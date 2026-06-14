from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from policychain.ingestion.normalizer import clean_policy_text


class PolicyLoadError(RuntimeError):
    """Raised when a policy file cannot be loaded safely."""


@dataclass
class LoadedPolicyFile:
    source_path: Path
    original_filename: str
    file_type: str
    text: str
    page_count: int | None = None


def read_policy_file(path: str | Path) -> LoadedPolicyFile:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Policy file does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        text, page_count = _read_pdf(file_path)
    elif suffix in {".md", ".txt"}:
        text = _read_text(file_path)
        page_count = None
    else:
        raise PolicyLoadError(f"Unsupported policy file type: {suffix or '<none>'}")

    text = clean_policy_text(text)
    if not text:
        raise PolicyLoadError(f"No extractable text found in policy file: {file_path}")

    return LoadedPolicyFile(
        source_path=file_path,
        original_filename=file_path.name,
        file_type=suffix.lstrip("."),
        text=text,
        page_count=page_count,
    )


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise PolicyLoadError(f"Unable to decode text policy file: {path}")


def _read_pdf(path: Path) -> tuple[str, int]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PolicyLoadError("PDF loading requires the 'pypdf' package") from exc

    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    except Exception as exc:  # pypdf raises several file-specific exception types.
        raise PolicyLoadError(f"Failed to extract PDF text from {path}: {exc}") from exc

    return "\n\n".join(pages), len(reader.pages)
