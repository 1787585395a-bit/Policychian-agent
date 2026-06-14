from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policychain.ingestion.pipeline import ingest_policy_file
from policychain.paths import SAMPLE_DB_PATH
from policychain.storage.sqlite_store import SQLitePolicyStore


DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "sample" / "raw"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "sample" / "policy_manifest.csv"
DEFAULT_DB = SAMPLE_DB_PATH


def ingest_sample_database(
    db_path: str | Path = DEFAULT_DB,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    reset: bool = False,
    max_chunk_chars: int = 1200,
) -> dict[str, Any]:
    db = Path(db_path)
    raw = Path(raw_dir)
    manifest = Path(manifest_path)

    if reset and db.exists():
        db.unlink()
    db.parent.mkdir(parents=True, exist_ok=True)

    policy_files = sorted(path for path in raw.iterdir() if path.suffix.lower() in {".pdf", ".md", ".txt"})
    if not policy_files:
        raise FileNotFoundError(f"No sample policy files found in {raw}")

    store = SQLitePolicyStore(db)
    ingested_policy_ids: list[str] = []
    try:
        for policy_file in policy_files:
            ingested_policy = ingest_policy_file(
                policy_file,
                manifest_path=manifest,
                max_chunk_chars=max_chunk_chars,
            )
            store.upsert_ingested_policy(ingested_policy)
            ingested_policy_ids.append(ingested_policy.metadata.policy_id)

        return {
            "db_path": str(db),
            "policy_ids": ingested_policy_ids,
            "policy_count": store.count_policies(),
            "chunk_count": store.count_chunks(),
            "fts_enabled": store.fts_enabled,
        }
    finally:
        store.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the sample PolicyChain SQLite database.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Output SQLite database path.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Directory containing sample policy files.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Sample policy manifest CSV path.")
    parser.add_argument("--reset", action="store_true", help="Delete the existing sample database before ingesting.")
    parser.add_argument("--max-chunk-chars", type=int, default=1200, help="Maximum chunk size used by ingestion.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = ingest_sample_database(
        db_path=args.db,
        raw_dir=args.raw_dir,
        manifest_path=args.manifest,
        reset=args.reset,
        max_chunk_chars=args.max_chunk_chars,
    )
    print(f"db_path={result['db_path']}")
    print(f"policy_count={result['policy_count']}")
    print(f"chunk_count={result['chunk_count']}")
    print(f"fts_enabled={result['fts_enabled']}")
    for policy_id in result["policy_ids"]:
        print(f"policy_id={policy_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
