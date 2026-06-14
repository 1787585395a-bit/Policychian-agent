from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PolicyMetadata:
    policy_id: str
    title: str
    document_number: str | None = None
    publish_date: str | None = None
    issuing_agencies: list[str] = field(default_factory=list)
    policy_level: str | None = None
    policy_type: str | None = None
    geographic_scope: str | None = None
    policy_status: str | None = None
    source_url: str | None = None
    original_filename: str | None = None
    normalized_filename: str | None = None
    file_hash: str | None = None
    file_type: str | None = None
    policy_domains: list[str] = field(default_factory=list)
    target_industries: list[str] = field(default_factory=list)
    target_entities: list[str] = field(default_factory=list)
    policy_tools: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    related_policy_ids: list[str] = field(default_factory=list)
    parent_policy_id: str | None = None
    implementation_policy_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyDocument:
    policy_id: str
    source_path: Path
    original_filename: str
    file_type: str
    file_hash: str
    text: str
    page_count: int | None = None

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_path"] = str(self.source_path)
        data["char_count"] = self.char_count
        return data


@dataclass
class PolicyChunk:
    chunk_id: str
    policy_id: str
    section_title: str
    section_index: int
    chunk_index: int
    page_start: int | None
    page_end: int | None
    char_start: int
    char_end: int
    previous_chunk_id: str | None
    next_chunk_id: str | None
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IngestedPolicy:
    metadata: PolicyMetadata
    document: PolicyDocument
    chunks: list[PolicyChunk]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "document": self.document.to_dict(),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
