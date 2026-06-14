from __future__ import annotations

import re
from dataclasses import dataclass

from policychain.ingestion.id_generator import generate_chunk_id
from policychain.ingestion.normalizer import clean_policy_text
from policychain.schemas.policy_schema import PolicyChunk


SECTION_RE = re.compile(
    r"^(第[一二三四五六七八九十百千万0-9]+章[\s\S]{0,60}|[一二三四五六七八九十百千万0-9]+、[\s\S]{1,60})$"
)
ARTICLE_RE = re.compile(r"(?<!^)(第[一二三四五六七八九十百千万0-9]+条)")


@dataclass
class TextSection:
    title: str
    index: int
    content: str


def chunk_policy_text(policy_id: str, text: str, max_chars: int = 1200) -> list[PolicyChunk]:
    cleaned = clean_policy_text(text)
    if not cleaned:
        return []

    sections = split_sections(cleaned)
    draft_chunks: list[PolicyChunk] = []
    search_from = 0

    for section in sections:
        units = split_policy_units(section.content)
        grouped_units = group_units(units, max_chars=max_chars)
        for chunk_index, content in enumerate(grouped_units, start=1):
            char_start = cleaned.find(content, search_from)
            if char_start == -1:
                char_start = cleaned.find(content)
            if char_start == -1:
                char_start = search_from
            char_end = char_start + len(content)
            search_from = char_end
            draft_chunks.append(
                PolicyChunk(
                    chunk_id=generate_chunk_id(policy_id, section.index, chunk_index),
                    policy_id=policy_id,
                    section_title=section.title,
                    section_index=section.index,
                    chunk_index=chunk_index,
                    page_start=None,
                    page_end=None,
                    char_start=char_start,
                    char_end=char_end,
                    previous_chunk_id=None,
                    next_chunk_id=None,
                    content=content,
                )
            )

    for index, chunk in enumerate(draft_chunks):
        chunk.previous_chunk_id = draft_chunks[index - 1].chunk_id if index > 0 else None
        chunk.next_chunk_id = draft_chunks[index + 1].chunk_id if index < len(draft_chunks) - 1 else None

    return draft_chunks


def split_sections(text: str) -> list[TextSection]:
    lines = text.splitlines()
    sections: list[TextSection] = []
    current_title = "正文"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if is_section_heading(stripped):
            if current_lines:
                sections.append(
                    TextSection(
                        title=current_title,
                        index=len(sections) + 1,
                        content="\n".join(current_lines).strip(),
                    )
                )
                current_lines = []
            current_title = stripped
            continue
        if stripped:
            current_lines.append(stripped)

    if current_lines:
        sections.append(
            TextSection(
                title=current_title,
                index=len(sections) + 1,
                content="\n".join(current_lines).strip(),
            )
        )

    if not sections:
        return [TextSection(title="正文", index=1, content=text)]
    return sections


def split_policy_units(text: str) -> list[str]:
    prepared = ARTICLE_RE.sub(r"\n\1", text)
    units = [unit.strip() for unit in re.split(r"\n+", prepared) if unit.strip()]
    if units:
        return units
    return [text.strip()] if text.strip() else []


def group_units(units: list[str], max_chars: int) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0

    for unit in units:
        if len(unit) > max_chars:
            if buffer:
                chunks.append("\n".join(buffer).strip())
                buffer = []
                buffer_len = 0
            chunks.extend(split_long_unit(unit, max_chars=max_chars))
            continue

        projected_len = buffer_len + len(unit) + (1 if buffer else 0)
        if buffer and projected_len > max_chars:
            chunks.append("\n".join(buffer).strip())
            buffer = [unit]
            buffer_len = len(unit)
        else:
            buffer.append(unit)
            buffer_len = projected_len

    if buffer:
        chunks.append("\n".join(buffer).strip())
    return chunks


def split_long_unit(unit: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[。；;.!?？])", unit)
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(buffer) + len(sentence) > max_chars and buffer:
            chunks.append(buffer.strip())
            buffer = sentence
        else:
            buffer += sentence
    if buffer:
        chunks.append(buffer.strip())
    return chunks or [unit[:max_chars]]


def is_section_heading(line: str) -> bool:
    if not line or len(line) > 80:
        return False
    return bool(SECTION_RE.match(line))
