from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policychain.mcp import diagnose_mcp_config, mcp_diagnostics_have_errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local PolicyChain MCP config without making live MCP calls.")
    parser.add_argument("--mcp-config", default=".mcp.local.json", help="Path to local MCP config.")
    parser.add_argument("--json", action="store_true", help="Print diagnostics as JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    diagnostics = diagnose_mcp_config(args.mcp_config, project_root=PROJECT_ROOT)
    if args.json:
        print(json.dumps([item.to_dict() for item in diagnostics], ensure_ascii=False, indent=2))
    else:
        for item in diagnostics:
            server = item.server_name or "config"
            print(f"[{item.status}] {server}.{item.check}: {item.message}")
    return 1 if mcp_diagnostics_have_errors(diagnostics) else 0


if __name__ == "__main__":
    raise SystemExit(main())
