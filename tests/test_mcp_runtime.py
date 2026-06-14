from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from policychain.mcp import (
    FakeMCPInvoker,
    MCPToolUnavailable,
    StdioMCPInvoker,
    cache_mcp_invoker,
    consume_mcp_invoker_errors,
    diagnose_mcp_config,
    load_mcp_config,
    mcp_diagnostics_have_errors,
    _mcp_child_env,
    unwrap_mcp_result,
)


class DummyTextContent:
    def __init__(self, text: str) -> None:
        self.text = text


class DummyCallResult:
    def __init__(self, content) -> None:
        self.content = content


class MCPRuntimeTests(unittest.TestCase):
    def test_load_mcp_example_config_expands_project_root(self) -> None:
        config = load_mcp_config(".mcp.example.json", project_root=Path.cwd())

        self.assertIn("web-search", config)
        self.assertIn("cn-financial", config)
        self.assertIn("cninfo", config)
        self.assertIn(str(Path.cwd()), config["cn-financial"].env["PYTHONPATH"])

    def test_stdio_invoker_missing_server_fails_clearly(self) -> None:
        invoker = StdioMCPInvoker(servers={})

        with self.assertRaisesRegex(MCPToolUnavailable, "not configured"):
            invoker.invoke("missing", "tool", {})

    def test_unwrap_mcp_result_accepts_json_text(self) -> None:
        result = DummyCallResult([DummyTextContent('{"results": [{"title": "ok"}]}')])

        self.assertEqual(unwrap_mcp_result(result), {"results": [{"title": "ok"}]})

    def test_unwrap_mcp_result_accepts_plain_text(self) -> None:
        result = DummyCallResult([DummyTextContent("plain response")])

        self.assertEqual(unwrap_mcp_result(result), {"content": "plain response"})

    def test_consume_mcp_invoker_errors_clears_errors(self) -> None:
        invoker = StdioMCPInvoker(servers={})
        invoker.errors.append("first error")

        self.assertEqual(consume_mcp_invoker_errors(invoker), ["first error"])
        self.assertEqual(consume_mcp_invoker_errors(invoker), [])

    def test_caching_invoker_caches_successful_calls_and_returns_copies(self) -> None:
        fake = FakeMCPInvoker({("server", "tool"): {"items": [{"title": "first"}]}})
        cached = cache_mcp_invoker(fake)

        first = cached.invoke("server", "tool", {"query": "政策"})
        first["items"][0]["title"] = "mutated"
        second = cached.invoke("server", "tool", {"query": "政策"})

        self.assertEqual(second["items"][0]["title"], "first")
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(cached.cache_info()["hits"], 1)
        self.assertEqual(cached.cache_info()["misses"], 1)

    def test_diagnose_mcp_config_reports_ok_for_existing_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pythonpath = root / "src"
            pythonpath.mkdir()
            config_path = root / "mcp.json"
            config_path.write_text(
                json.dumps(
                    {
                        "mcpServers": {
                            "local": {
                                "type": "stdio",
                                "command": "python",
                                "args": ["-m", "example"],
                                "env": {"PYTHONPATH": str(pythonpath)},
                                "cwd": str(root),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            diagnostics = diagnose_mcp_config(config_path, project_root=root)

        self.assertFalse(mcp_diagnostics_have_errors(diagnostics), [item.to_dict() for item in diagnostics])

    def test_diagnose_mcp_config_reports_missing_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "mcp.json"
            config_path.write_text(
                json.dumps({"mcpServers": {"bad": {"type": "stdio", "command": str(root / "missing.exe")}}}),
                encoding="utf-8",
            )

            diagnostics = diagnose_mcp_config(config_path, project_root=root)

        self.assertTrue(mcp_diagnostics_have_errors(diagnostics))
        self.assertTrue(any(item.check == "command" and item.status == "error" for item in diagnostics))

    def test_mcp_child_env_drops_system_proxy_by_default(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "HTTP_PROXY": "http://127.0.0.1:9",
                "HTTPS_PROXY": "http://127.0.0.1:9",
                "ALL_PROXY": "http://127.0.0.1:9",
            },
            clear=True,
        ):
            env = _mcp_child_env({"MODE": "stdio"})

        self.assertEqual(env["MODE"], "stdio")
        self.assertNotIn("HTTP_PROXY", env)
        self.assertNotIn("HTTPS_PROXY", env)
        self.assertNotIn("ALL_PROXY", env)

    def test_mcp_child_env_can_opt_into_system_proxy(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "HTTP_PROXY": "http://127.0.0.1:7890",
                "POLICYCHAIN_MCP_USE_SYSTEM_PROXY": "1",
            },
            clear=True,
        ):
            env = _mcp_child_env({})

        self.assertEqual(env["HTTP_PROXY"], "http://127.0.0.1:7890")


if __name__ == "__main__":
    unittest.main()
