from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_COMPANY_DATA = Path("data/sample/mock_companies.json")


def load_mock_companies(path: str | Path = DEFAULT_COMPANY_DATA) -> list[dict[str, Any]]:
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Company mock data does not exist: {data_path}")
    return json.loads(data_path.read_text(encoding="utf-8"))


def search_company_information(
    industry_segment: str,
    keywords: list[str] | None = None,
    companies: list[dict[str, Any]] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search mock company profiles by industry segment and keyword overlap."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    records = companies if companies is not None else load_mock_companies()
    query_terms = [industry_segment, *(keywords or [])]
    scored: list[tuple[int, dict[str, Any]]] = []
    for record in records:
        haystack = " ".join(
            [
                str(record.get("industry_segment") or ""),
                str(record.get("matched_business") or ""),
                " ".join(record.get("business_keywords") or []),
                str(record.get("business_evidence") or ""),
            ]
        )
        score = sum(1 for term in query_terms if term and term in haystack)
        if score > 0:
            scored.append((score, record))

    scored.sort(key=lambda item: (-item[0], item[1].get("company_name", "")))
    return [record for _, record in scored[:top_k]]


def read_company_source(company_record: dict[str, Any]) -> dict[str, Any]:
    """Return the source fields used as company-business evidence."""

    return {
        "source_name": company_record.get("source_name"),
        "source_url": company_record.get("source_url"),
        "text": company_record.get("business_evidence"),
        "data_date": company_record.get("data_date"),
    }
