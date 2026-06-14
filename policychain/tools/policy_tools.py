from __future__ import annotations

from typing import Any

from policychain.storage.sqlite_store import SQLitePolicyStore


def search_policy(
    store: SQLitePolicyStore,
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Search ingested policy chunks with a small keyword scorer."""

    return store.search_policy(query=query, filters=filters, top_k=top_k)


def get_policy_metadata(
    store: SQLitePolicyStore,
    policy_ids: list[str],
) -> list[dict[str, Any]]:
    """Return policy identity metadata for known policy IDs."""

    return [metadata.to_dict() for metadata in store.get_policy_metadata(policy_ids)]


def read_policy_content(
    store: SQLitePolicyStore,
    policy_id: str,
    chunk_ids: list[str] | None = None,
    include_neighbors: bool = False,
) -> dict[str, Any]:
    """Read a full policy or selected chunks, optionally adding neighboring chunks."""

    if chunk_ids:
        selected_ids = set(chunk_ids)
        if include_neighbors:
            for chunk in store.get_chunks(chunk_ids=chunk_ids):
                if chunk.previous_chunk_id:
                    selected_ids.add(chunk.previous_chunk_id)
                if chunk.next_chunk_id:
                    selected_ids.add(chunk.next_chunk_id)
        chunks = store.get_chunks(chunk_ids=sorted(selected_ids))
    else:
        chunks = store.get_chunks(policy_id=policy_id)

    chunks = [chunk for chunk in chunks if chunk.policy_id == policy_id]
    metadata = store.get_policy_metadata([policy_id])
    document = store.get_policy_document(policy_id)

    return {
        "policy_id": policy_id,
        "metadata": metadata[0].to_dict() if metadata else None,
        "document": document.to_dict() if document and not chunk_ids else None,
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
