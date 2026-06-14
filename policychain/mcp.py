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


PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
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
                    self._invoke_async(self.servers[server_name], tool_name, arguments),
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

    async def _invoke_async(self, server: MCPServerConfig, tool_name: str, arguments: dict[str, Any]) -> Any:
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
        return unwrap_mcp_result(result)


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
        return result
    if isinstance(result, str):
        return _parse_text_payload(result)

    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured

    content = getattr(result, "content", None)
    if content is not None:
        items = [_unwrap_content_item(item) for item in content]
        if len(items) == 1:
            return items[0]
        return items

    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return {"content": str(result)}


def consume_mcp_invoker_errors(invoker: MCPToolInvoker | None) -> list[str]:
    consumer = getattr(invoker, "consume_errors", None)
    if callable(consumer):
        return list(consumer())
    return []


def cache_mcp_invoker(invoker: MCPToolInvoker, max_entries: int = 256) -> CachingMCPInvoker:
    return CachingMCPInvoker(inner=invoker, max_entries=max_entries)


def is_unavailable_invoker(invoker: MCPToolInvoker | None) -> bool:
    return invoker is None or isinstance(invoker, UnavailableMCPInvoker)


def _mcp_child_env(server_env: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.update(server_env)
    if not _truthy_env(env.get(MCP_USE_SYSTEM_PROXY_ENV)):
        for key in PROXY_ENV_KEYS:
            env.pop(key, None)
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
        return parsed
    return {"content": parsed}


def _truthy_env(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _mcp_cache_key(server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
    encoded_args = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    return json.dumps([server_name, tool_name, encoded_args], ensure_ascii=False)


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
