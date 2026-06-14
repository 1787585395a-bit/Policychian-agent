from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policychain.ingestion.id_generator import compute_file_hash
from policychain.ingestion.pipeline import ingest_policy_file
from policychain.paths import FULL_DB_PATH
from policychain.storage.sqlite_store import SQLitePolicyStore


DEFAULT_SOURCE_DIR = Path(r"D:\Code\人工智能政策文件")
DEFAULT_MANIFEST = DEFAULT_SOURCE_DIR / "政策文件清单.csv"
DEFAULT_DB = FULL_DB_PATH
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def ingest_policy_directory(
    db_path: str | Path = DEFAULT_DB,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    manifest_path: str | Path | None = DEFAULT_MANIFEST,
    reset: bool = False,
    max_chunk_chars: int = 1200,
    limit: int | None = None,
    skip_existing_hash: bool = True,
) -> dict[str, Any]:
    db = Path(db_path)
    source = Path(source_dir)
    manifest = Path(manifest_path) if manifest_path else None

    if not source.is_dir():
        raise FileNotFoundError(f"Policy source directory does not exist: {source}")
    if manifest and not manifest.is_file():
        raise FileNotFoundError(f"Policy manifest does not exist: {manifest}")
    if reset and db.exists():
        db.unlink()
    db.parent.mkdir(parents=True, exist_ok=True)

    policy_files = sorted(path for path in source.iterdir() if path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        policy_files = policy_files[:limit]
    if not policy_files:
        raise FileNotFoundError(f"No policy files found in {source}")

    store = SQLitePolicyStore(db)
    ingested_policy_ids: list[str] = []
    skipped_files: list[str] = []
    try:
        for policy_file in policy_files:
            file_hash = compute_file_hash(policy_file)
            existing = store.find_policy_by_hash(file_hash)
            if existing and skip_existing_hash:
                skipped_files.append(policy_file.name)
                continue

            ingested_policy = ingest_policy_file(
                policy_file,
                manifest_path=manifest,
                max_chunk_chars=max_chunk_chars,
            )
            store.upsert_ingested_policy(ingested_policy)
            ingested_policy_ids.append(ingested_policy.metadata.policy_id)

        return {
            "db_path": str(db),
            "source_dir": str(source),
            "manifest_path": str(manifest) if manifest else None,
            "discovered_file_count": len(policy_files),
            "ingested_policy_ids": ingested_policy_ids,
            "ingested_count": len(ingested_policy_ids),
            "skipped_files": skipped_files,
            "skipped_count": len(skipped_files),
            "policy_count": store.count_policies(),
            "chunk_count": store.count_chunks(),
            "fts_enabled": store.fts_enabled,
        }
    finally:
        store.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PolicyChain SQLite database from a policy directory.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Output SQLite database path.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="Directory containing policy files.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Policy manifest CSV path.")
    parser.add_argument("--no-manifest", action="store_true", help="Ingest without a manifest CSV.")
    parser.add_argument("--reset", action="store_true", help="Delete the existing database before ingesting.")
    parser.add_argument("--max-chunk-chars", type=int, default=1200, help="Maximum chunk size used by ingestion.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick smoke runs.")
    parser.add_argument(
        "--reingest-existing-hash",
        action="store_true",
        help="Reingest files even when the same file hash already exists.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = ingest_policy_directory(
        db_path=args.db,
        source_dir=args.source_dir,
        manifest_path=None if args.no_manifest else args.manifest,
        reset=args.reset,
        max_chunk_chars=args.max_chunk_chars,
        limit=args.limit,
        skip_existing_hash=not args.reingest_existing_hash,
    )
    print(f"db_path={result['db_path']}")
    print(f"source_dir={result['source_dir']}")
    print(f"manifest_path={result['manifest_path']}")
    print(f"discovered_file_count={result['discovered_file_count']}")
    print(f"ingested_count={result['ingested_count']}")
    print(f"skipped_count={result['skipped_count']}")
    print(f"policy_count={result['policy_count']}")
    print(f"chunk_count={result['chunk_count']}")
    print(f"fts_enabled={result['fts_enabled']}")
    for policy_id in result["ingested_policy_ids"]:
        print(f"policy_id={policy_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
