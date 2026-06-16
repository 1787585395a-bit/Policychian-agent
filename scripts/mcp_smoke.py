from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policychain.mcp import StdioMCPInvoker
from policychain.tools.mcp_tools import (
    CNFINANCIAL_SERVER,
    OPEN_WEBSEARCH_SEARCH_TOOL,
    OPEN_WEBSEARCH_SERVER,
)


def run_smoke(
    config_path: str | Path = ".mcp.local.json",
    timeout_seconds: float = 60,
) -> dict[str, Any]:
    invoker = StdioMCPInvoker.from_config_file(config_path, timeout_seconds=timeout_seconds, project_root=PROJECT_ROOT)
    result: dict[str, Any] = {}

    web = invoker.invoke(
        OPEN_WEBSEARCH_SERVER,
        OPEN_WEBSEARCH_SEARCH_TOOL,
        {"query": "生成式人工智能 服务管理 官方解读", "limit": 1},
    )
    result["web_search"] = _count_payload(web)

    industry = invoker.invoke(CNFINANCIAL_SERVER, "get_industry_list", {})
    result["cnfinancial_industry_list"] = _count_payload(industry)

    news = invoker.invoke(
        CNFINANCIAL_SERVER,
        "search_news",
        {"keyword": "人工智能", "query": "人工智能", "limit": 1},
    )
    result["cnfinancial_search_news"] = _count_payload(news)

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check configured PolicyChain MCP stdio servers.")
    parser.add_argument("--mcp-config", default=".mcp.local.json", help="Path to local MCP config.")
    parser.add_argument("--timeout", type=float, default=60, help="Timeout seconds per MCP call.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_smoke(
        config_path=args.mcp_config,
        timeout_seconds=args.timeout,
    )
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


def _count_payload(payload: Any) -> int:
    if payload is None:
        return 0
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("results", "data", "items", "reports", "stocks", "companies"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
        return 1
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
