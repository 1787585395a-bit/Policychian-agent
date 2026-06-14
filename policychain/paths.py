from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DB_PATH = PROJECT_ROOT / "data" / "processed" / "policychain.sqlite"
FULL_DB_PATH = PROJECT_ROOT / "data" / "processed" / "policychain_full.sqlite"


def resolve_default_db_path() -> Path:
    configured = os.environ.get("POLICYCHAIN_DB")
    if configured:
        return Path(configured)
    if FULL_DB_PATH.exists():
        return FULL_DB_PATH
    return SAMPLE_DB_PATH


def is_full_db_path(path: str | Path) -> bool:
    return Path(path).resolve() == FULL_DB_PATH.resolve()
