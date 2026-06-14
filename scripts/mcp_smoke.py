from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policychain.mcp import StdioMCPInvoker
from policychain.tools.mcp_tools import (
    CNFINANCIAL_SERVER,
    CNINFO_DOWNLOAD_ANNUAL_REPORTS_TOOL,
    CNINFO_QUERY_ANNUAL_REPORTS_TOOL,
    CNINFO_SERVER,
    OPEN_WEBSEARCH_SEARCH_TOOL,
    OPEN_WEBSEARCH_SERVER,
    select_recent_annual_reports,
)


def run_smoke(
    config_path: str | Path = ".mcp.local.json",
    timeout_seconds: float = 60,
    stock_code: str = "000888",
    download_cninfo: bool = False,
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

    reports = invoker.invoke(
        CNINFO_SERVER,
        CNINFO_QUERY_ANNUAL_REPORTS_TOOL,
        {"stock_code": stock_code},
    )
    recent_reports = select_recent_annual_reports(reports, limit=2, stock_code=stock_code)
    result["cninfo_annual_reports"] = len(recent_reports)
    result["cninfo_recent_years"] = [_report_year_hint(report) for report in recent_reports]

    if download_cninfo and recent_reports:
        year = _report_year_hint(recent_reports[0])
        downloaded = invoker.invoke(
            CNINFO_SERVER,
            CNINFO_DOWNLOAD_ANNUAL_REPORTS_TOOL,
            {"stock_code": stock_code, "year": year},
        )
        result["cninfo_download"] = _count_payload(downloaded)

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check configured PolicyChain MCP stdio servers.")
    parser.add_argument("--mcp-config", default=".mcp.local.json", help="Path to local MCP config.")
    parser.add_argument("--timeout", type=float, default=60, help="Timeout seconds per MCP call.")
    parser.add_argument("--stock-code", default="000888", help="Stock code used for CNINFO annual report smoke.")
    parser.add_argument("--download-cninfo", action="store_true", help="Also download the latest annual report PDF.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_smoke(
        config_path=args.mcp_config,
        timeout_seconds=args.timeout,
        stock_code=args.stock_code,
        download_cninfo=args.download_cninfo,
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


def _report_year_hint(report: dict[str, Any]) -> Any:
    for key in ("year", "report_year", "title", "report_title", "announcementTitle", "announcement_title"):
        value = report.get(key)
        if value:
            match = re.search(r"(20\d{2}|19\d{2})", str(value))
            if match:
                return int(match.group(1))
    return None


if __name__ == "__main__":
    raise SystemExit(main())
