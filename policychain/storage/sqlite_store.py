from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

from policychain.schemas.policy_schema import (
    IngestedPolicy,
    PolicyChunk,
    PolicyDocument,
    PolicyMetadata,
)


LIST_METADATA_FIELDS = {
    "issuing_agencies",
    "policy_domains",
    "target_industries",
    "target_entities",
    "policy_tools",
    "keywords",
    "related_policy_ids",
    "implementation_policy_ids",
}

QUERY_MARKERS = (
    "生成式人工智能",
    "人工智能",
    "服务提供者",
    "提供者",
    "管理要求",
    "管理",
    "办法",
    "政策",
    "备案",
    "监督",
)


class SQLitePolicyStore:
    """Small SQLite-backed store for ingested policy metadata, documents, and chunks."""

    def __init__(self, db_path: str | Path = ":memory:", enable_fts: bool = True) -> None:
        self.db_path = str(db_path)
        self._enable_fts = enable_fts
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        if self.db_path != ":memory:":
            self._connection.execute("PRAGMA journal_mode = OFF")
            self._connection.execute("PRAGMA synchronous = OFF")
        self._connection.execute("PRAGMA foreign_keys = ON")
        self.fts_enabled = False
        self.initialize_schema()

    def close(self) -> None:
        self._connection.close()

    def initialize_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS policies (
                policy_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                document_number TEXT,
                publish_date TEXT,
                issuing_agencies TEXT NOT NULL,
                policy_level TEXT,
                policy_type TEXT,
                geographic_scope TEXT,
                policy_status TEXT,
                source_url TEXT,
                original_filename TEXT,
                normalized_filename TEXT,
                file_hash TEXT,
                file_type TEXT,
                policy_domains TEXT NOT NULL,
                target_industries TEXT NOT NULL,
                target_entities TEXT NOT NULL,
                policy_tools TEXT NOT NULL,
                keywords TEXT NOT NULL,
                related_policy_ids TEXT NOT NULL,
                parent_policy_id TEXT,
                implementation_policy_ids TEXT NOT NULL,
                source_path TEXT NOT NULL,
                document_text TEXT NOT NULL,
                page_count INTEGER,
                char_count INTEGER NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_policies_file_hash
            ON policies(file_hash)
            WHERE file_hash IS NOT NULL;

            CREATE INDEX IF NOT EXISTS idx_policies_publish_date
            ON policies(publish_date);

            CREATE INDEX IF NOT EXISTS idx_policies_policy_level
            ON policies(policy_level);

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                policy_id TEXT NOT NULL,
                section_title TEXT NOT NULL,
                section_index INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                page_start INTEGER,
                page_end INTEGER,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL,
                previous_chunk_id TEXT,
                next_chunk_id TEXT,
                content TEXT NOT NULL,
                FOREIGN KEY(policy_id) REFERENCES policies(policy_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_policy_id
            ON chunks(policy_id);
            """
        )
        if self._enable_fts:
            self._initialize_fts()
        self._connection.commit()

    def _initialize_fts(self) -> None:
        try:
            self._connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(
                    chunk_id UNINDEXED,
                    policy_id UNINDEXED,
                    title,
                    section_title,
                    content,
                    tokenize='trigram'
                )
                """
            )
            self.fts_enabled = True
            self.rebuild_fts_index()
        except sqlite3.Error:
            self.fts_enabled = False

    def upsert_ingested_policy(self, ingested_policy: IngestedPolicy) -> None:
        metadata = ingested_policy.metadata
        document = ingested_policy.document
        policy_row = _metadata_to_storage_row(metadata)
        policy_row.update(
            {
                "source_path": str(document.source_path),
                "document_text": document.text,
                "page_count": document.page_count,
                "char_count": document.char_count,
            }
        )

        policy_columns = list(policy_row.keys())
        placeholders = ", ".join([f":{column}" for column in policy_columns])
        assignments = ", ".join([f"{column}=excluded.{column}" for column in policy_columns if column != "policy_id"])
        self._connection.execute(
            f"""
            INSERT INTO policies ({", ".join(policy_columns)})
            VALUES ({placeholders})
            ON CONFLICT(policy_id) DO UPDATE SET {assignments}
            """,
            policy_row,
        )

        self._delete_chunks_for_policy(metadata.policy_id)
        self._connection.executemany(
            """
            INSERT INTO chunks (
                chunk_id,
                policy_id,
                section_title,
                section_index,
                chunk_index,
                page_start,
                page_end,
                char_start,
                char_end,
                previous_chunk_id,
                next_chunk_id,
                content
            )
            VALUES (
                :chunk_id,
                :policy_id,
                :section_title,
                :section_index,
                :chunk_index,
                :page_start,
                :page_end,
                :char_start,
                :char_end,
                :previous_chunk_id,
                :next_chunk_id,
                :content
            )
            """,
            [chunk.to_dict() for chunk in ingested_policy.chunks],
        )
        self._sync_fts_for_policy(ingested_policy)
        self._connection.commit()

    def delete_policy(self, policy_id: str, commit: bool = True) -> None:
        self._delete_chunks_for_policy(policy_id)
        self._connection.execute("DELETE FROM policies WHERE policy_id = ?", (policy_id,))
        if commit:
            self._connection.commit()

    def _delete_chunks_for_policy(self, policy_id: str) -> None:
        self._connection.execute("DELETE FROM chunks WHERE policy_id = ?", (policy_id,))
        if self.fts_enabled:
            self._connection.execute("DELETE FROM chunks_fts WHERE policy_id = ?", (policy_id,))

    def rebuild_fts_index(self) -> None:
        if not self.fts_enabled:
            return
        self._connection.execute("DELETE FROM chunks_fts")
        self._connection.execute(
            """
            INSERT INTO chunks_fts (
                chunk_id,
                policy_id,
                title,
                section_title,
                content
            )
            SELECT
                c.chunk_id,
                c.policy_id,
                p.title,
                c.section_title,
                c.content
            FROM chunks c
            JOIN policies p ON p.policy_id = c.policy_id
            """
        )

    def count_policies(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS count FROM policies").fetchone()
        return int(row["count"])

    def count_chunks(self) -> int:
        row = self._connection.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"])

    def find_policy_by_hash(self, file_hash: str) -> PolicyMetadata | None:
        row = self._connection.execute(
            "SELECT * FROM policies WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        return _metadata_from_row(row) if row else None

    def get_policy_metadata(self, policy_ids: Iterable[str]) -> list[PolicyMetadata]:
        ids = list(policy_ids)
        if not ids:
            return []

        placeholders = ", ".join("?" for _ in ids)
        rows = self._connection.execute(
            f"SELECT * FROM policies WHERE policy_id IN ({placeholders}) ORDER BY policy_id",
            ids,
        ).fetchall()
        return [_metadata_from_row(row) for row in rows]

    def get_policy_document(self, policy_id: str) -> PolicyDocument | None:
        row = self._connection.execute(
            "SELECT * FROM policies WHERE policy_id = ?",
            (policy_id,),
        ).fetchone()
        if row is None:
            return None
        return PolicyDocument(
            policy_id=row["policy_id"],
            source_path=Path(row["source_path"]),
            original_filename=row["original_filename"] or "",
            file_type=row["file_type"] or "",
            file_hash=row["file_hash"] or "",
            text=row["document_text"],
            page_count=row["page_count"],
        )

    def get_chunks(
        self,
        policy_id: str | None = None,
        chunk_ids: Iterable[str] | None = None,
    ) -> list[PolicyChunk]:
        ids = list(chunk_ids or [])
        if ids:
            placeholders = ", ".join("?" for _ in ids)
            rows = self._connection.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders}) ORDER BY section_index, chunk_index",
                ids,
            ).fetchall()
        elif policy_id:
            rows = self._connection.execute(
                "SELECT * FROM chunks WHERE policy_id = ? ORDER BY section_index, chunk_index",
                (policy_id,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT * FROM chunks ORDER BY policy_id, section_index, chunk_index"
            ).fetchall()

        return [_chunk_from_row(row) for row in rows]

    def search_policy(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        terms = _query_terms(query)
        if not terms:
            return []

        if self.fts_enabled:
            try:
                return self._search_policy_fts(query=query, filters=filters or {}, top_k=top_k, terms=terms)
            except sqlite3.Error:
                return self._search_policy_python(terms=terms, filters=filters or {}, top_k=top_k)

        return self._search_policy_python(terms=terms, filters=filters or {}, top_k=top_k)

    def _search_policy_python(
        self,
        terms: list[str],
        filters: dict[str, Any],
        top_k: int,
    ) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT
                c.chunk_id,
                c.policy_id,
                c.section_title,
                c.content,
                p.title,
                p.publish_date,
                p.issuing_agencies,
                p.policy_level,
                p.policy_type,
                p.source_url
            FROM chunks c
            JOIN policies p ON p.policy_id = c.policy_id
            """
        ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            if not _matches_filters(row, filters or {}):
                continue
            score = _score_row(row, terms)
            if score <= 0:
                continue
            results.append(
                {
                    "policy_id": row["policy_id"],
                    "chunk_id": row["chunk_id"],
                    "title": row["title"],
                    "publish_date": row["publish_date"],
                    "agency": ", ".join(_json_loads(row["issuing_agencies"], default=[])),
                    "matched_text": _snippet(row["content"], terms),
                    "score": score,
                    "source_url": row["source_url"],
                    "section_title": row["section_title"],
                }
            )

        results.sort(key=lambda item: (-item["score"], item["policy_id"], item["chunk_id"]))
        return results[:top_k]

    def _search_policy_fts(
        self,
        query: str,
        filters: dict[str, Any],
        top_k: int,
        terms: list[str],
    ) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT
                c.chunk_id,
                c.policy_id,
                c.section_title,
                c.content,
                p.title,
                p.publish_date,
                p.issuing_agencies,
                p.policy_level,
                p.policy_type,
                p.source_url,
                bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            JOIN policies p ON p.policy_id = c.policy_id
            WHERE chunks_fts MATCH ?
            """,
            (_fts_query(query),),
        ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            if not _matches_filters(row, filters):
                continue
            score = _score_row(row, terms)
            rank = row["rank"]
            if score <= 0:
                score = max(0.1, 10.0 - abs(float(rank or 0.0)))
            results.append(
                {
                    "policy_id": row["policy_id"],
                    "chunk_id": row["chunk_id"],
                    "title": row["title"],
                    "publish_date": row["publish_date"],
                    "agency": ", ".join(_json_loads(row["issuing_agencies"], default=[])),
                    "matched_text": _snippet(row["content"], terms),
                    "score": score,
                    "source_url": row["source_url"],
                    "section_title": row["section_title"],
                }
            )

        results.sort(key=lambda item: (-item["score"], item["policy_id"], item["chunk_id"]))
        return results[:top_k]

    def _sync_fts_for_policy(self, ingested_policy: IngestedPolicy) -> None:
        if not self.fts_enabled:
            return
        self._connection.execute(
            "DELETE FROM chunks_fts WHERE policy_id = ?",
            (ingested_policy.metadata.policy_id,),
        )
        self._connection.executemany(
            """
            INSERT INTO chunks_fts (
                chunk_id,
                policy_id,
                title,
                section_title,
                content
            )
            VALUES (
                :chunk_id,
                :policy_id,
                :title,
                :section_title,
                :content
            )
            """,
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "policy_id": chunk.policy_id,
                    "title": ingested_policy.metadata.title,
                    "section_title": chunk.section_title,
                    "content": chunk.content,
                }
                for chunk in ingested_policy.chunks
            ],
        )


def _metadata_to_storage_row(metadata: PolicyMetadata) -> dict[str, Any]:
    data = metadata.to_dict()
    for field_name in LIST_METADATA_FIELDS:
        data[field_name] = json.dumps(data.get(field_name) or [], ensure_ascii=False)
    return data


def _metadata_from_row(row: sqlite3.Row) -> PolicyMetadata:
    values: dict[str, Any] = {}
    for field in fields(PolicyMetadata):
        value = row[field.name]
        if field.name in LIST_METADATA_FIELDS:
            value = _json_loads(value, default=[])
        values[field.name] = value
    return PolicyMetadata(**values)


def _chunk_from_row(row: sqlite3.Row) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=row["chunk_id"],
        policy_id=row["policy_id"],
        section_title=row["section_title"],
        section_index=row["section_index"],
        chunk_index=row["chunk_index"],
        page_start=row["page_start"],
        page_end=row["page_end"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        previous_chunk_id=row["previous_chunk_id"],
        next_chunk_id=row["next_chunk_id"],
        content=row["content"],
    )


def _query_terms(query: str) -> list[str]:
    query = query.strip().lower()
    terms = [term for term in re_split_query(query) if term]
    if len(terms) == 1 and _contains_cjk(query):
        terms.extend(marker for marker in QUERY_MARKERS if marker in query)
    return _unique_terms(terms)


def _fts_query(query: str) -> str:
    terms = [term for term in _query_terms(query) if len(term) >= 3]
    if not terms:
        terms = [query.strip()]
    # Trigram tokenization supports Chinese substring lookup without external segmenters.
    return " OR ".join(f'"{term.replace(chr(34), chr(34) + chr(34))}"' for term in terms)


def re_split_query(query: str) -> list[str]:
    return [term for term in re.split(r"\s+", query) if term]


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def _unique_terms(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _score_row(row: sqlite3.Row, terms: list[str]) -> float:
    title = (row["title"] or "").lower()
    section = (row["section_title"] or "").lower()
    content = (row["content"] or "").lower()
    score = 0.0
    for term in terms:
        if term in title:
            score += 5.0
        if term in section:
            score += 2.0
        score += min(content.count(term), 5)
    return score


def _matches_filters(row: sqlite3.Row, filters: dict[str, Any]) -> bool:
    year_from = filters.get("year_from")
    year_to = filters.get("year_to")
    publish_year = _publish_year(row["publish_date"])
    if year_from is not None and (publish_year is None or publish_year < int(year_from)):
        return False
    if year_to is not None and (publish_year is None or publish_year > int(year_to)):
        return False

    agency = filters.get("agency")
    if agency and agency not in ", ".join(_json_loads(row["issuing_agencies"], default=[])):
        return False

    policy_level = filters.get("policy_level")
    if policy_level and row["policy_level"] != policy_level:
        return False

    policy_type = filters.get("policy_type") or filters.get("industry")
    if policy_type and row["policy_type"] != policy_type:
        return False

    return True


def _publish_year(value: str | None) -> int | None:
    if not value or len(value) < 4:
        return None
    try:
        return int(value[:4])
    except ValueError:
        return None


def _snippet(content: str, terms: list[str], radius: int = 80) -> str:
    lower_content = content.lower()
    positions = [lower_content.find(term) for term in terms if lower_content.find(term) >= 0]
    if not positions:
        return content[: radius * 2].strip()
    start = max(min(positions) - radius, 0)
    end = min(min(positions) + radius, len(content))
    return content[start:end].strip()


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
