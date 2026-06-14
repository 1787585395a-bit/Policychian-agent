from __future__ import annotations

from pathlib import Path

from policychain.ingestion.chunker import chunk_policy_text
from policychain.ingestion.id_generator import compute_file_hash
from policychain.ingestion.loaders import read_policy_file
from policychain.ingestion.metadata_extractor import extract_metadata
from policychain.schemas.policy_schema import IngestedPolicy, PolicyDocument


class IngestionError(RuntimeError):
    """Raised when a policy file cannot complete the ingestion pipeline."""


def ingest_policy_file(
    path: str | Path,
    manifest_path: str | Path | None = None,
    max_chunk_chars: int = 1200,
) -> IngestedPolicy:
    file_hash = compute_file_hash(path)
    loaded_file = read_policy_file(path)
    metadata = extract_metadata(
        loaded_file=loaded_file,
        file_hash=file_hash,
        manifest_path=manifest_path,
    )
    document = PolicyDocument(
        policy_id=metadata.policy_id,
        source_path=loaded_file.source_path,
        original_filename=loaded_file.original_filename,
        file_type=loaded_file.file_type,
        file_hash=file_hash,
        text=loaded_file.text,
        page_count=loaded_file.page_count,
    )
    chunks = chunk_policy_text(
        policy_id=metadata.policy_id,
        text=loaded_file.text,
        max_chars=max_chunk_chars,
    )
    if not chunks:
        raise IngestionError(f"No chunks generated for policy file: {path}")

    return IngestedPolicy(metadata=metadata, document=document, chunks=chunks)
