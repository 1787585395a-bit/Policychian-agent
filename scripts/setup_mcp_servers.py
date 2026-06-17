from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CNFINANCIAL_REPO = "https://github.com/ccq1/cn-financial-mcp"
DEFAULT_EXTERNAL_DIR = PROJECT_ROOT / "external" / "mcp"
DEFAULT_CNFINANCIAL_DIR = DEFAULT_EXTERNAL_DIR / "cn-financial-mcp"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / ".mcp.local.json"


def build_mcp_config(
    cnfinancial_dir: Path = DEFAULT_CNFINANCIAL_DIR,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    return {
        "mcpServers": {
            "web-search": _npx_server(
                "open-websearch@latest",
                env={"MODE": "stdio", "DEFAULT_SEARCH_ENGINE": "bing", "ALLOWED_SEARCH_ENGINES": "bing", "SEARCH_MODE": "request"},
            ),
            "cn-financial": {
                "type": "stdio",
                "command": python_executable,
                "args": ["-m", "cn_financial_mcp"],
                "env": {
                    "PYTHONPATH": str(cnfinancial_dir / "src"),
                },
                "cwd": str(cnfinancial_dir),
            },
        }
    }


def setup_mcp_servers(
    external_dir: Path = DEFAULT_EXTERNAL_DIR,
    config_path: Path = DEFAULT_CONFIG_PATH,
    skip_clone: bool = False,
    skip_install: bool = False,
) -> dict[str, Any]:
    node_version = _run_capture(["node", "-v"])
    npm_version = _run_capture(["npm", "-v"])
    python_version = _run_capture([sys.executable, "--version"])

    cnfinancial_dir = external_dir / "cn-financial-mcp"
    external_dir.mkdir(parents=True, exist_ok=True)
    if not skip_clone:
        _clone_or_update_cnfinancial(cnfinancial_dir)
    if not skip_install:
        _install_cnfinancial(cnfinancial_dir)

    config = build_mcp_config(
        cnfinancial_dir=cnfinancial_dir,
        python_executable=sys.executable,
    )
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "node_version": node_version,
        "npm_version": npm_version,
        "python_version": python_version,
        "cnfinancial_dir": str(cnfinancial_dir),
        "config_path": str(config_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local stdio MCP server config for PolicyChain.")
    parser.add_argument("--external-dir", default=str(DEFAULT_EXTERNAL_DIR), help="Directory for external MCP repositories.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Output .mcp.local.json path.")
    parser.add_argument("--skip-clone", action="store_true", help="Do not clone or update cn-financial-mcp.")
    parser.add_argument("--skip-install", action="store_true", help="Do not pip install cn-financial-mcp.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = setup_mcp_servers(
        external_dir=Path(args.external_dir),
        config_path=Path(args.config),
        skip_clone=args.skip_clone,
        skip_install=args.skip_install,
    )
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


def _npx_server(package: str, env: dict[str, str]) -> dict[str, Any]:
    if platform.system().lower().startswith("win"):
        command = "cmd"
        args = ["/c", "npx", "-y", package]
    else:
        command = "npx"
        args = ["-y", package]
    return {
        "type": "stdio",
        "command": command,
        "args": args,
        "env": env,
    }


def _clone_or_update_cnfinancial(target: Path) -> None:
    if not target.exists():
        _run_checked(["git", "clone", CNFINANCIAL_REPO, str(target)])
        return
    git_dir = target / ".git"
    if not git_dir.is_dir():
        raise FileExistsError(f"Target exists but is not a git repository: {target}")
    _run_checked(["git", "-C", str(target), "pull", "--ff-only"])


def _install_cnfinancial(cnfinancial_dir: Path) -> None:
    try:
        _run_checked([sys.executable, "-m", "pip", "install", "-e", str(cnfinancial_dir)])
    except subprocess.CalledProcessError as exc:
        requirements_path = cnfinancial_dir / "requirements.txt"
        if not requirements_path.is_file():
            raise RuntimeError(
                "Failed to install cn-financial-mcp with pip, and requirements.txt "
                "fallback is unavailable."
            ) from exc
        print(
            "pip editable install failed; falling back to requirements.txt dependency install. "
            "The MCP server will run through PYTHONPATH=src.",
            file=sys.stderr,
        )
        _run_checked([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])


def _run_checked(command: list[str]) -> None:
    subprocess.run(_resolved_command(command), cwd=PROJECT_ROOT, check=True)


def _run_capture(command: list[str]) -> str:
    result = subprocess.run(_resolved_command(command), cwd=PROJECT_ROOT, check=True, text=True, capture_output=True)
    return (result.stdout or result.stderr).strip()


def _resolved_command(command: list[str]) -> list[str]:
    if not command:
        return command
    executable = shutil.which(command[0])
    if executable:
        return [executable, *command[1:]]
    return command


if __name__ == "__main__":
    raise SystemExit(main())
