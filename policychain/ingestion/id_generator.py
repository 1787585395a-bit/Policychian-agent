from __future__ import annotations

import hashlib
import re
from pathlib import Path


class IDGenerationError(ValueError):
    """Raised when an identifier cannot be generated from valid inputs."""


def compute_file_hash(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Policy file does not exist: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_policy_id(
    year: int | str,
    agency_code: str,
    sequence: int | str,
    region_code: str | None = None,
) -> str:
    year_text = str(year)
    if not re.fullmatch(r"\d{4}", year_text):
        raise IDGenerationError(f"Policy year must be four digits: {year!r}")

    agency = _clean_code(agency_code, "agency_code")
    seq = int(sequence)
    if seq <= 0:
        raise IDGenerationError("Policy sequence must be a positive integer")

    if region_code:
        region = _clean_code(region_code, "region_code")
        return f"POL-{year_text}-{region}-{agency}-{seq:04d}"
    return f"POL-{year_text}-{agency}-{seq:04d}"


def generate_chunk_id(policy_id: str, section_index: int, chunk_index: int) -> str:
    if not policy_id.startswith("POL-"):
        raise IDGenerationError(f"Invalid policy_id: {policy_id!r}")
    if section_index <= 0:
        raise IDGenerationError("section_index must be positive")
    if chunk_index <= 0:
        raise IDGenerationError("chunk_index must be positive")
    return f"{policy_id}-S{section_index:02d}-C{chunk_index:03d}"


def _clean_code(value: str, field_name: str) -> str:
    code = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,12}", code):
        raise IDGenerationError(f"{field_name} must contain 2-12 ASCII letters/digits: {value!r}")
    return code
