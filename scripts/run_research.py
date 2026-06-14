from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policychain.graph import run_llm_policy_research_workflow, run_policy_research_workflow
from policychain.llm import LLMClient
from policychain.mcp import MCPToolInvoker, StdioMCPInvoker, cache_mcp_invoker
from policychain.paths import FULL_DB_PATH, SAMPLE_DB_PATH, is_full_db_path, resolve_default_db_path
from policychain.storage import SQLitePolicyStore
from scripts.ingest_sample import ingest_sample_database


DEFAULT_QUERY = "生成式人工智能服务提供者有哪些管理要求"


def run_research(
    query: str = DEFAULT_QUERY,
    db_path: str | Path | None = None,
    ensure_sample_db: bool = True,
    rebuild_sample_db: bool = False,
    output_path: str | Path | None = None,
    use_llm: bool = False,
    llm_client: LLMClient | None = None,
    mcp_invoker: MCPToolInvoker | None = None,
    skip_annual_reports: bool = False,
    progress_callback: Callable[[int, str, str], None] | None = None,
) -> str:
    db = Path(db_path) if db_path else resolve_default_db_path()
    if ensure_sample_db and (rebuild_sample_db or not db.exists()):
        if is_full_db_path(db):
            raise FileNotFoundError(
                "Full policy database does not exist. Build it with "
                "`python scripts/ingest_policy_dir.py --reset` before running against the full database."
            )
        ingest_sample_database(db_path=db, reset=rebuild_sample_db)

    store = SQLitePolicyStore(db)
    try:
        if use_llm:
            state = run_llm_policy_research_workflow(
                query,
                store,
                llm_client=llm_client,
                mcp_invoker=mcp_invoker,
                use_annual_reports=not skip_annual_reports,
                progress_callback=progress_callback,
            )
        else:
            state = run_policy_research_workflow(
                query,
                store,
                mcp_invoker=mcp_invoker,
                use_annual_reports=not skip_annual_reports,
                progress_callback=progress_callback,
            )
    finally:
        store.close()

    report = state.final_report
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the sample PolicyChain research workflow.")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Policy research question.")
    parser.add_argument("--db", default=None, help="SQLite policy database path.")
    parser.add_argument("--sample-db", action="store_true", help="Use the one-policy sample database.")
    parser.add_argument("--full-db", action="store_true", help="Use the full local policy database.")
    parser.add_argument("--rebuild-sample-db", action="store_true", help="Rebuild the sample database before running.")
    parser.add_argument("--no-ingest", action="store_true", help="Do not auto-build a missing sample database.")
    parser.add_argument("--out", default=None, help="Optional Markdown report output path.")
    parser.add_argument("--llm", action="store_true", help="Use the optional LLM-backed workflow.")
    parser.add_argument("--mcp", action="store_true", help="Use real stdio MCP tools from a local MCP config.")
    parser.add_argument("--mcp-config", default=".mcp.local.json", help="Path to local MCP config used with --mcp.")
    parser.add_argument("--mcp-timeout", type=float, default=60, help="Timeout seconds per MCP tool call.")
    parser.add_argument("--no-mcp-cache", action="store_true", help="Disable in-process cache for repeated MCP calls.")
    parser.add_argument("--skip-annual-reports", action="store_true", help="Skip CNINFO annual report query/download during company matching.")
    args = parser.parse_args(argv)
    if args.sample_db and args.full_db:
        parser.error("--sample-db and --full-db cannot be used together.")
    if args.db and (args.sample_db or args.full_db):
        parser.error("--db cannot be combined with --sample-db or --full-db.")
    if args.rebuild_sample_db and args.full_db:
        parser.error("--rebuild-sample-db cannot be used with --full-db.")
    return args


def _selected_db_path(args: argparse.Namespace) -> Path | None:
    if args.sample_db:
        return SAMPLE_DB_PATH
    if args.full_db:
        return FULL_DB_PATH
    if args.db:
        return Path(args.db)
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mcp_invoker = None
    if args.mcp:
        stdio_invoker = StdioMCPInvoker.from_config_file(
            args.mcp_config,
            timeout_seconds=args.mcp_timeout,
            project_root=PROJECT_ROOT,
        )
        mcp_invoker = stdio_invoker if args.no_mcp_cache else cache_mcp_invoker(stdio_invoker)
    report = run_research(
        query=args.query,
        db_path=_selected_db_path(args),
        ensure_sample_db=not args.no_ingest,
        rebuild_sample_db=args.rebuild_sample_db,
        output_path=args.out,
        use_llm=args.llm,
        mcp_invoker=mcp_invoker,
        skip_annual_reports=args.skip_annual_reports,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
