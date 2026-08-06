from __future__ import annotations

import asyncio
from collections import OrderedDict
from copy import deepcopy
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class MCPToolInvoker(Protocol):
    """Small boundary for calling externally configured MCP tools."""

    def invoke(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool and return its raw payload."""


class MCPToolUnavailable(RuntimeError):
    """Raised when no real MCP runtime is configured for a requested tool."""


class MCPToolError(RuntimeError):
    """Raised when a configured MCP tool fails at runtime."""


class MCPToolCircuitOpen(MCPToolUnavailable):
    """Raised when a run-local MCP circuit is open and a call is skipped."""


PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "GIT_HTTP_PROXY",
    "GIT_HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "git_http_proxy",
    "git_https_proxy",
)
MCP_USE_SYSTEM_PROXY_ENV = "POLICYCHAIN_MCP_USE_SYSTEM_PROXY"


@dataclass(frozen=True)
class MCPServerConfig:
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    server_type: str = "stdio"


@dataclass(frozen=True)
class MCPDiagnostic:
    server_name: str
    check: str
    status: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "server_name": self.server_name,
            "check": self.check,
            "status": self.status,
            "message": self.message,
        }


class UnavailableMCPInvoker:
    """Default invoker used when external MCP servers are not configured."""

    def invoke(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        raise MCPToolUnavailable(f"MCP tool is not configured: {server_name}.{tool_name}")


@dataclass
class FakeMCPInvoker:
    """Deterministic test invoker keyed by (server_name, tool_name)."""

    responses: dict[tuple[str, str], Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def invoke(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append(
            {
                "server_name": server_name,
                "tool_name": tool_name,
                "arguments": dict(arguments),
            }
        )
        key = (server_name, tool_name)
        if key not in self.responses:
            raise MCPToolUnavailable(f"No fake MCP response registered for {server_name}.{tool_name}")
        response = self.responses[key]
        if callable(response):
            return response(server_name=server_name, tool_name=tool_name, arguments=dict(arguments))
        return response

    def consume_errors(self) -> list[str]:
        errors = list(self.errors)
        self.errors.clear()
        return errors


MCP_CATALOG_TOOLS = {"get_industry_list", "get_concept_list"}
MCP_SERVICE_FATAL_PATTERNS = (
    "winerror 5",
    "access is denied",
    "permission denied",
    "createprocess",
    "failed to create process",
)
MCP_TOOL_TRANSPORT_PATTERNS = (
    "remotedisconnected",
    "remote end closed connection",
    "connection reset",
    "connection aborted",
    "broken pipe",
)


@dataclass
class RuntimeMCPInvoker:
    """Run-local MCP health, catalog cache, and explainable circuit breaker."""

    inner: MCPToolInvoker
    failure_threshold: int = 2
    errors: list[str] = field(default_factory=list)
    _tool_health: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    _server_health: dict[str, dict[str, Any]] = field(default_factory=dict)
    _catalog_cache: dict[tuple[str, str], Any] = field(default_factory=dict)
    _preflight_servers: set[str] = field(default_factory=set)
    _query_budgets: dict[tuple[str, str, str], int] = field(default_factory=dict)
    _last_call: dict[str, Any] = field(default_factory=dict)

    def preflight(self, server_name: str) -> tuple[dict[str, Any], bool]:
        """Inspect configuration once without issuing a speculative remote call."""

        first_check = server_name not in self._preflight_servers
        self._preflight_servers.add(server_name)
        health = self._server_health.setdefault(
            server_name,
            {
                "status": "ready",
                "consecutive_failures": 0,
                "circuit_open": False,
                "circuit_reason": "",
            },
        )
        configured, reason = _runtime_server_configured(self.inner, server_name)
        if not configured:
            health.update(
                {
                    "status": "unavailable",
                    "circuit_open": True,
                    "circuit_reason": reason,
                }
            )
        return deepcopy(health), first_check

    def invoke(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        server_health, _first_check = self.preflight(server_name)
        tool_key = (server_name, tool_name)
        tool_health = self._tool_health.setdefault(
            tool_key,
            {
                "status": "unknown",
                "consecutive_failures": 0,
                "circuit_open": False,
                "circuit_reason": "",
            },
        )
        self._last_call = {
            "server_name": server_name,
            "tool_name": tool_name,
            "cache_hit": False,
            "skipped": False,
            "circuit_open": False,
            "circuit_scope": "",
            "failure_count": int(tool_health.get("consecutive_failures") or 0),
            "server_status": str(server_health.get("status") or "unknown"),
            "tool_status": str(tool_health.get("status") or "unknown"),
        }

        if server_health.get("circuit_open"):
            self._raise_circuit_open(
                server_name,
                tool_name,
                scope="service",
                reason=str(server_health.get("circuit_reason") or "service unavailable"),
            )
        if tool_health.get("circuit_open"):
            self._raise_circuit_open(
                server_name,
                tool_name,
                scope="tool",
                reason=str(tool_health.get("circuit_reason") or "tool unavailable"),
            )

        if tool_name in MCP_CATALOG_TOOLS and tool_key in self._catalog_cache:
            cached = deepcopy(self._catalog_cache[tool_key])
            status = "empty" if _runtime_payload_count(cached) == 0 else "ok"
            self._last_call.update(
                {
                    "cache_hit": True,
                    "status": status,
                    "server_status": "ok",
                    "tool_status": status,
                    "failure_count": 0,
                }
            )
            return cached

        try:
            result = self.inner.invoke(server_name, tool_name, arguments)
            payload_error = mcp_payload_error_message(result, server_name=server_name, tool_name=tool_name)
            if payload_error:
                raise MCPToolError(payload_error)
        except MCPToolCircuitOpen:
            raise
        except MCPToolUnavailable as exc:
            self._register_failure(server_name, tool_name, str(exc), unavailable=True)
            raise
        except MCPToolError as exc:
            self._register_failure(server_name, tool_name, str(exc), unavailable=False)
            raise
        except Exception as exc:
            message = f"MCP tool failed: {server_name}.{tool_name}: {exc}"
            self._register_failure(server_name, tool_name, message, unavailable=False)
            raise MCPToolError(message) from exc

        status = "empty" if _runtime_payload_count(result) == 0 else "ok"
        self._register_success(server_name, tool_name, status)
        if tool_name in MCP_CATALOG_TOOLS:
            self._catalog_cache[tool_key] = deepcopy(result)
        return result

    def call_metadata(self) -> dict[str, Any]:
        return deepcopy(self._last_call)

    def reserve_query_budget(
        self,
        server_name: str,
        tool_name: str,
        scope_id: str,
        *,
        limit: int,
    ) -> tuple[bool, int]:
        """Reserve one actual remote attempt in a run-local scoped budget."""

        key = (server_name, tool_name, scope_id)
        used = int(self._query_budgets.get(key) or 0)
        if used >= max(int(limit), 0):
            return False, used
        used += 1
        self._query_budgets[key] = used
        return True, used

    def release_query_budget(self, server_name: str, tool_name: str, scope_id: str) -> int:
        """Release a reservation when a circuit prevented an actual attempt."""

        key = (server_name, tool_name, scope_id)
        used = max(int(self._query_budgets.get(key) or 0) - 1, 0)
        if used:
            self._query_budgets[key] = used
        else:
            self._query_budgets.pop(key, None)
        return used

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "servers": deepcopy(self._server_health),
            "tools": {
                f"{server_name}.{tool_name}": deepcopy(health)
                for (server_name, tool_name), health in self._tool_health.items()
            },
            "catalog_entries": len(self._catalog_cache),
            "preflight_servers": sorted(self._preflight_servers),
            "query_budgets": {
                f"{server_name}.{tool_name}.{scope_id}": used
                for (server_name, tool_name, scope_id), used in self._query_budgets.items()
            },
        }

    def server_is_unavailable(self, server_name: str) -> bool:
        health, _first_check = self.preflight(server_name)
        return bool(health.get("circuit_open") and health.get("status") == "unavailable")

    def consume_errors(self) -> list[str]:
        errors = list(self.errors)
        self.errors.clear()
        consumer = getattr(self.inner, "consume_errors", None)
        if callable(consumer):
            errors.extend(str(item) for item in consumer())
        return list(dict.fromkeys(error for error in errors if error))

    def close(self) -> None:
        closer = getattr(self.inner, "close", None)
        if callable(closer):
            closer()

    def _register_success(self, server_name: str, tool_name: str, status: str) -> None:
        server_health = self._server_health.setdefault(server_name, {})
        server_health.update(
            {
                "status": "ok",
                "consecutive_failures": 0,
                "circuit_open": False,
                "circuit_reason": "",
            }
        )
        tool_health = self._tool_health.setdefault((server_name, tool_name), {})
        tool_health.update(
            {
                "status": status,
                "consecutive_failures": 0,
                "circuit_open": False,
                "circuit_reason": "",
            }
        )
        self._last_call.update(
            {
                "status": status,
                "server_status": "ok",
                "tool_status": status,
                "failure_count": 0,
            }
        )

    def _register_failure(self, server_name: str, tool_name: str, message: str, *, unavailable: bool) -> None:
        normalized = message.lower()
        service_fatal = any(pattern in normalized for pattern in MCP_SERVICE_FATAL_PATTERNS)
        transport_failure = any(pattern in normalized for pattern in MCP_TOOL_TRANSPORT_PATTERNS)
        tool_health = self._tool_health.setdefault((server_name, tool_name), {})
        failure_count = int(tool_health.get("consecutive_failures") or 0) + 1
        open_tool = unavailable or transport_failure or failure_count >= max(int(self.failure_threshold), 1)
        status = "unavailable" if unavailable else "error"
        tool_health.update(
            {
                "status": status,
                "consecutive_failures": failure_count,
                "circuit_open": bool(open_tool and not service_fatal),
                "circuit_reason": message if open_tool and not service_fatal else "",
            }
        )
        server_health = self._server_health.setdefault(server_name, {})
        if service_fatal:
            server_health.update(
                {
                    "status": "unavailable",
                    "consecutive_failures": int(server_health.get("consecutive_failures") or 0) + 1,
                    "circuit_open": True,
                    "circuit_reason": message,
                }
            )
        else:
            server_health.update(
                {
                    "status": "degraded",
                    "consecutive_failures": int(server_health.get("consecutive_failures") or 0) + 1,
                    "circuit_open": False,
                }
            )
        if message and message not in self.errors:
            self.errors.append(message)
        self._last_call.update(
            {
                "status": status,
                "error": message,
                "server_status": str(server_health.get("status") or "unknown"),
                "tool_status": status,
                "failure_count": failure_count,
                "circuit_open": bool(service_fatal or (open_tool and not service_fatal)),
                "circuit_scope": "service" if service_fatal else ("tool" if open_tool else ""),
            }
        )

    def _raise_circuit_open(self, server_name: str, tool_name: str, *, scope: str, reason: str) -> None:
        message = f"MCP {scope} circuit open; skipped {server_name}.{tool_name}: {reason}"
        self._last_call.update(
            {
                "status": "unavailable",
                "error": message,
                "skipped": True,
                "circuit_open": True,
                "circuit_scope": scope,
                "server_status": "unavailable" if scope == "service" else self._last_call.get("server_status", "degraded"),
                "tool_status": "unavailable",
            }
        )
        raise MCPToolCircuitOpen(message)


@dataclass
class CachingMCPInvoker:
    """Small in-process cache wrapper for deterministic repeated MCP calls."""

    inner: MCPToolInvoker
    max_entries: int = 256
    hits: int = 0
    misses: int = 0
    _cache: OrderedDict[str, Any] = field(default_factory=OrderedDict)

    def invoke(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        key = _mcp_cache_key(server_name, tool_name, arguments)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return deepcopy(self._cache[key])

        self.misses += 1
        result = self.inner.invoke(server_name, tool_name, arguments)
        if self.max_entries > 0:
            self._cache[key] = deepcopy(result)
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
        return result

    def cache_info(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self._cache),
            "max_entries": self.max_entries,
        }

    def consume_errors(self) -> list[str]:
        consumer = getattr(self.inner, "consume_errors", None)
        if callable(consumer):
            return list(consumer())
        return []

    def close(self) -> None:
        closer = getattr(self.inner, "close", None)
        if callable(closer):
            closer()


@dataclass
class StdioMCPInvoker:
    """Synchronous wrapper around stdio MCP servers configured in a local JSON file."""

    servers: dict[str, MCPServerConfig]
    timeout_seconds: float = 60.0
    project_root: Path = field(default_factory=lambda: Path.cwd())
    errors: list[str] = field(default_factory=list)

    @classmethod
    def from_config_file(
        cls,
        config_path: str | Path = ".mcp.local.json",
        timeout_seconds: float = 60.0,
        project_root: str | Path | None = None,
    ) -> "StdioMCPInvoker":
        root = Path(project_root) if project_root else Path.cwd()
        config = load_mcp_config(config_path, project_root=root)
        return cls(servers=config, timeout_seconds=timeout_seconds, project_root=root)

    def invoke(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> Any:
        if server_name not in self.servers:
            raise MCPToolUnavailable(f"MCP server is not configured: {server_name}")
        try:
            return asyncio.run(
                asyncio.wait_for(
                    self._invoke_async(server_name, self.servers[server_name], tool_name, arguments),
                    timeout=self.timeout_seconds,
                )
            )
        except MCPToolUnavailable:
            raise
        except TimeoutError as exc:
            message = f"MCP tool timed out after {self.timeout_seconds}s: {server_name}.{tool_name}"
            self.errors.append(message)
            raise MCPToolError(message) from exc
        except RuntimeError as exc:
            if "asyncio.run() cannot be called from a running event loop" in str(exc):
                message = "StdioMCPInvoker cannot be used from an already-running asyncio event loop"
                self.errors.append(message)
                raise MCPToolError(message) from exc
            message = f"MCP tool failed: {server_name}.{tool_name}: {exc}"
            self.errors.append(message)
            raise MCPToolError(message) from exc
        except Exception as exc:
            message = f"MCP tool failed: {server_name}.{tool_name}: {exc}"
            self.errors.append(message)
            raise MCPToolError(message) from exc

    def consume_errors(self) -> list[str]:
        errors = list(self.errors)
        self.errors.clear()
        return errors

    async def _invoke_async(self, server_name: str, server: MCPServerConfig, tool_name: str, arguments: dict[str, Any]) -> Any:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise MCPToolUnavailable(
                "Python package `mcp` is not installed. Run `python -m pip install -r requirements.txt`."
            ) from exc

        if server.server_type != "stdio":
            raise MCPToolUnavailable(f"Only stdio MCP servers are supported by StdioMCPInvoker: {server.server_type}")

        env = _mcp_child_env(server.env)
        params = _stdio_server_parameters(
            StdioServerParameters=StdioServerParameters,
            command=server.command,
            args=server.args,
            env=env,
            cwd=server.cwd,
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
        payload = unwrap_mcp_result(result)
        error_message = mcp_payload_error_message(payload, server_name=server_name, tool_name=tool_name)
        if error_message:
            self.errors.append(error_message)
            raise MCPToolError(error_message)
        return payload


def load_mcp_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, MCPServerConfig]:
    root = Path(project_root) if project_root else Path.cwd()
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"MCP config does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict) or not servers:
        raise ValueError("MCP config must contain a non-empty `mcpServers` object")

    parsed: dict[str, MCPServerConfig] = {}
    for name, raw_server in servers.items():
        if not isinstance(raw_server, dict):
            raise ValueError(f"MCP server config must be an object: {name}")
        command = raw_server.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"MCP server config requires `command`: {name}")
        args = raw_server.get("args") or []
        env = raw_server.get("env") or {}
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError(f"MCP server config `args` must be a string list: {name}")
        if not isinstance(env, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items()):
            raise ValueError(f"MCP server config `env` must be a string map: {name}")
        cwd = raw_server.get("cwd")
        server_type = str(raw_server.get("type") or raw_server.get("transport") or "stdio")
        parsed[name] = MCPServerConfig(
            command=_expand_config_value(command, root),
            args=[_expand_config_value(arg, root) for arg in args],
            env={key: _expand_config_value(value, root) for key, value in env.items()},
            cwd=_expand_config_value(cwd, root) if isinstance(cwd, str) else None,
            server_type=server_type,
        )
    return parsed


def unwrap_mcp_result(result: Any) -> Any:
    if result is None:
        return []
    if isinstance(result, (dict, list)):
        return _normalize_nested_payload(result)
    if isinstance(result, str):
        return _parse_text_payload(result)

    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return _normalize_nested_payload(structured)

    content = getattr(result, "content", None)
    if content is not None:
        items = [_unwrap_content_item(item) for item in content]
        if len(items) == 1:
            return items[0]
        return items

    if hasattr(result, "model_dump"):
        return _normalize_nested_payload(result.model_dump())
    if hasattr(result, "dict"):
        return _normalize_nested_payload(result.dict())
    return {"content": str(result)}


def mcp_payload_error_message(payload: Any, server_name: str, tool_name: str) -> str | None:
    """Return a clear error message when an MCP payload reports failure."""

    prefix = f"{server_name}.{tool_name}".strip(".")
    for item in _payload_error_items(payload):
        if item.get("error") is True or str(item.get("status") or "").lower() == "error":
            message = (
                item.get("message")
                or item.get("error_message")
                or item.get("detail")
                or item.get("content")
                or "MCP tool returned an error payload"
            )
            return f"MCP tool returned error payload: {prefix}: {message}"
    return None


def consume_mcp_invoker_errors(invoker: MCPToolInvoker | None) -> list[str]:
    runtime = getattr(invoker, "_policychain_runtime_invoker", None)
    if isinstance(runtime, RuntimeMCPInvoker):
        return runtime.consume_errors()
    consumer = getattr(invoker, "consume_errors", None)
    if callable(consumer):
        return list(consumer())
    return []


def cache_mcp_invoker(invoker: MCPToolInvoker, max_entries: int = 256) -> CachingMCPInvoker:
    return CachingMCPInvoker(inner=invoker, max_entries=max_entries)


def runtime_mcp_invoker(invoker: MCPToolInvoker) -> RuntimeMCPInvoker:
    """Return the single resilient wrapper associated with this run invoker."""

    if isinstance(invoker, RuntimeMCPInvoker):
        return invoker
    existing = getattr(invoker, "_policychain_runtime_invoker", None)
    if isinstance(existing, RuntimeMCPInvoker):
        return existing
    runtime = RuntimeMCPInvoker(inner=invoker)
    try:
        setattr(invoker, "_policychain_runtime_invoker", runtime)
    except (AttributeError, TypeError):
        pass
    return runtime


def mcp_runtime_snapshot(invoker: MCPToolInvoker | None) -> dict[str, Any]:
    if invoker is None:
        return {
            "servers": {},
            "tools": {},
            "catalog_entries": 0,
            "preflight_servers": [],
            "query_budgets": {},
            "status": "unavailable",
        }
    return runtime_mcp_invoker(invoker).status_snapshot()


def mcp_server_is_unavailable(invoker: MCPToolInvoker | None, server_name: str) -> bool:
    if invoker is None or is_unavailable_invoker(invoker):
        return True
    return runtime_mcp_invoker(invoker).server_is_unavailable(server_name)


def is_unavailable_invoker(invoker: MCPToolInvoker | None) -> bool:
    if invoker is None or isinstance(invoker, UnavailableMCPInvoker):
        return True
    if isinstance(invoker, RuntimeMCPInvoker):
        return is_unavailable_invoker(invoker.inner)
    if isinstance(invoker, CachingMCPInvoker):
        return is_unavailable_invoker(invoker.inner)
    return False


def _mcp_child_env(server_env: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(server_env)
    if not _truthy_env(env.get(MCP_USE_SYSTEM_PROXY_ENV)):
        for key in PROXY_ENV_KEYS:
            env.pop(key, None)
        env["NO_PROXY"] = "*"
        env["no_proxy"] = "*"
    return env


def diagnose_mcp_config(
    config_path: str | Path,
    project_root: str | Path | None = None,
) -> list[MCPDiagnostic]:
    try:
        config = load_mcp_config(config_path, project_root=project_root)
    except Exception as exc:
        return [MCPDiagnostic(server_name="", check="config", status="error", message=str(exc))]

    diagnostics: list[MCPDiagnostic] = []
    for server_name, server in config.items():
        diagnostics.append(_diagnose_server_type(server_name, server))
        diagnostics.append(_diagnose_command(server_name, server.command))
        if server.cwd:
            diagnostics.append(_diagnose_cwd(server_name, server.cwd))
        for key, value in server.env.items():
            diagnostics.extend(_diagnose_env_path(server_name, key, value))
    return diagnostics


def mcp_diagnostics_have_errors(diagnostics: list[MCPDiagnostic]) -> bool:
    return any(item.status == "error" for item in diagnostics)


def _stdio_server_parameters(
    StdioServerParameters: Any,
    command: str,
    args: list[str],
    env: dict[str, str],
    cwd: str | None,
) -> Any:
    try:
        return StdioServerParameters(command=command, args=args, env=env, cwd=cwd)
    except TypeError:
        return StdioServerParameters(command=command, args=args, env=env)


def _unwrap_content_item(item: Any) -> Any:
    text = getattr(item, "text", None)
    if text is not None:
        return _parse_text_payload(text)
    if isinstance(item, (dict, list, str)):
        return unwrap_mcp_result(item)
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "dict"):
        return item.dict()
    return {"content": str(item)}


def _parse_text_payload(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {"content": stripped}
    if isinstance(parsed, (dict, list)):
        return _normalize_nested_payload(parsed)
    return {"content": parsed}


def _normalize_nested_payload(payload: Any) -> Any:
    if isinstance(payload, list):
        return [_normalize_nested_payload(item) for item in payload]
    if not isinstance(payload, dict):
        return payload

    normalized = {key: _normalize_nested_payload(value) for key, value in payload.items()}
    for key in ("result", "content", "text"):
        value = normalized.get(key)
        if not isinstance(value, str):
            continue
        parsed = _try_parse_json_text(value)
        if parsed is not None:
            if len(normalized) == 1:
                return parsed
            normalized[key] = parsed
    return normalized


def _try_parse_json_text(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return _normalize_nested_payload(parsed)


def _payload_error_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        items: list[dict[str, Any]] = [payload]
        for key in ("results", "data", "items", "reports", "announcements", "stocks", "companies", "result"):
            value = payload.get(key)
            if isinstance(value, dict):
                items.extend(_payload_error_items(value))
            elif isinstance(value, list):
                items.extend(_payload_error_items(value))
        return items
    if isinstance(payload, list):
        items = []
        for item in payload:
            items.extend(_payload_error_items(item))
        return items
    return []


def _truthy_env(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _mcp_cache_key(server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
    encoded_args = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    return json.dumps([server_name, tool_name, encoded_args], ensure_ascii=False)


def _runtime_server_configured(invoker: MCPToolInvoker, server_name: str) -> tuple[bool, str]:
    if isinstance(invoker, RuntimeMCPInvoker):
        return _runtime_server_configured(invoker.inner, server_name)
    if isinstance(invoker, CachingMCPInvoker):
        return _runtime_server_configured(invoker.inner, server_name)
    if isinstance(invoker, UnavailableMCPInvoker):
        return False, f"MCP server is not configured: {server_name}"
    if isinstance(invoker, StdioMCPInvoker) and server_name not in invoker.servers:
        return False, f"MCP server is not configured: {server_name}"
    return True, ""


def _runtime_payload_count(payload: Any) -> int:
    if payload is None:
        return 0
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("results", "result", "data", "items", "reports", "announcements", "stocks", "companies"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
            if isinstance(value, dict):
                return 1 if value else 0
        return 1 if payload else 0
    return 1


def _diagnose_server_type(server_name: str, server: MCPServerConfig) -> MCPDiagnostic:
    if server.server_type == "stdio":
        return MCPDiagnostic(server_name, "type", "ok", "stdio transport")
    return MCPDiagnostic(server_name, "type", "error", f"unsupported transport: {server.server_type}")


def _diagnose_command(server_name: str, command: str) -> MCPDiagnostic:
    command_path = Path(command)
    if command_path.is_absolute() or any(separator in command for separator in ("/", "\\")):
        if command_path.exists():
            return MCPDiagnostic(server_name, "command", "ok", command)
        return MCPDiagnostic(server_name, "command", "error", f"command path does not exist: {command}")
    resolved = shutil.which(command)
    if resolved:
        return MCPDiagnostic(server_name, "command", "ok", resolved)
    return MCPDiagnostic(server_name, "command", "error", f"command is not on PATH: {command}")


def _diagnose_cwd(server_name: str, cwd: str) -> MCPDiagnostic:
    path = Path(cwd)
    if path.is_dir():
        return MCPDiagnostic(server_name, "cwd", "ok", cwd)
    return MCPDiagnostic(server_name, "cwd", "error", f"cwd does not exist: {cwd}")


def _diagnose_env_path(server_name: str, key: str, value: str) -> list[MCPDiagnostic]:
    if key.upper() != "PYTHONPATH":
        return []
    diagnostics: list[MCPDiagnostic] = []
    for path_text in [part for part in value.split(os.pathsep) if part]:
        path = Path(path_text)
        if path.exists():
            diagnostics.append(MCPDiagnostic(server_name, f"env.{key}", "ok", path_text))
        else:
            diagnostics.append(MCPDiagnostic(server_name, f"env.{key}", "error", f"path does not exist: {path_text}"))
    return diagnostics


def _expand_config_value(value: str, project_root: Path) -> str:
    return (
        value.replace("${PROJECT_ROOT}", str(project_root))
        .replace("<PROJECT_ROOT>", str(project_root))
        .replace("{PROJECT_ROOT}", str(project_root))
    )
