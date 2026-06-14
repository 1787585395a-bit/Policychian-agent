from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from policychain.ingestion.pipeline import ingest_policy_file
from policychain.storage.sqlite_store import SQLitePolicyStore


SAMPLE_PDF_PATH = Path("data/sample/raw/48_国家_2023_生成式人工智能服务管理暂行办法.pdf")
SAMPLE_MANIFEST_PATH = Path("data/sample/policy_manifest.csv")


def build_sample_store(max_chunk_chars: int = 900) -> SQLitePolicyStore:
    ingested_policy = ingest_policy_file(
        SAMPLE_PDF_PATH,
        manifest_path=SAMPLE_MANIFEST_PATH,
        max_chunk_chars=max_chunk_chars,
    )
    store = SQLitePolicyStore(":memory:")
    store.upsert_ingested_policy(ingested_policy)
    return store


def artifact_db_path(name: str) -> Path:
    directory = Path("artifacts/test-results")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}_{uuid4().hex}.sqlite"
