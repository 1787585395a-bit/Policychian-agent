from __future__ import annotations

import re


def clean_policy_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t\u3000]+", " ", normalized)
    normalized = re.sub(r"\n[ \t]+", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def safe_filename_part(value: str, max_length: int = 80) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\r\n]+', "_", value).strip(" _")
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned[:max_length] or "untitled"
