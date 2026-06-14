from __future__ import annotations

from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from io import BytesIO
import re
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pypdf import PdfReader

from policychain.ingestion.chunker import group_units, split_policy_units, split_sections


MIN_POLICY_TEXT_CHARS = 120
MIN_EXTRACTED_POLICY_CHARS = 80
URL_TIMEOUT_SECONDS = 30
USER_AGENT = "PolicyChain/0.1 (+local research tool)"
POLICY_TEXT_MARKERS = (
    "第一条",
    "第二条",
    "第三条",
    "应当",
    "不得",
    "办法",
    "通知",
    "意见",
    "规定",
    "实施",
    "施行",
    "主管部门",
    "国务院",
    "办公厅",
    "国家",
    "制定本",
    "文号",
)
ERROR_PAGE_MARKERS = (
    "404",
    "403",
    "not found",
    "forbidden",
    "access denied",
    "页面不存在",
    "访问受限",
    "验证码",
    "登录后",
    "请登录",
)


class SourcePolicyError(RuntimeError):
    """Raised when a user-provided policy input cannot be read."""


def is_url_input(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def looks_like_policy_text(value: str) -> bool:
    text = _compact_text(value)
    if len(text) < 30:
        return False
    marker_count = _policy_marker_count(text)
    if len(text) >= MIN_POLICY_TEXT_CHARS and marker_count >= 2:
        return True
    return "\n" in text and marker_count >= 2


def build_source_policy_from_user_input(value: str) -> dict[str, Any]:
    stripped = value.strip()
    if not stripped:
        raise SourcePolicyError("Policy input is empty")
    if is_url_input(stripped):
        return build_source_policy_from_url(stripped)
    if looks_like_policy_text(stripped):
        return build_source_policy_from_text(stripped, raw_input=stripped, input_type="text")
    raise SourcePolicyError("Input is too short to treat as policy text")


def build_source_policy_from_url(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=URL_TIMEOUT_SECONDS) as response:
            payload = response.read()
            content_type = response.headers.get("Content-Type", "")
    except (OSError, URLError) as exc:
        raise SourcePolicyError(f"Failed to fetch policy URL: {url}: {exc}") from exc

    if _is_pdf_url(url, content_type):
        text = _pdf_text(payload)
        title = _title_from_url(url)
    elif "text/plain" in content_type:
        text = _decode_bytes(payload, content_type)
        title = _title_from_text(text) or _title_from_url(url)
    else:
        html = _decode_bytes(payload, content_type)
        title, text = _html_text(html)
        title = title or _title_from_url(url)

    clean_text = _compact_text(text)
    if not clean_text:
        raise SourcePolicyError(f"Fetched policy URL but extracted no readable text: {url}")
    validate_policy_text(clean_text, title=title, source_url=url)
    return build_source_policy_from_text(clean_text, raw_input=url, input_type="url", source_url=url, title=title)


def build_source_policy_from_text(
    text: str,
    raw_input: str,
    input_type: str,
    source_url: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    clean_text = _compact_text(text)
    if not clean_text:
        raise SourcePolicyError("Policy text is empty after cleanup")
    digest = sha256(clean_text.encode("utf-8")).hexdigest()
    policy_id = f"INPUT-{digest[:8].upper()}"
    title = _best_title(title, clean_text) or policy_id
    chunks = _chunk_source_policy(policy_id, clean_text)
    if not chunks:
        raise SourcePolicyError("Policy text could not be split into readable chunks")
    metadata = {
        "policy_id": policy_id,
        "title": title,
        "document_number": _document_number(clean_text),
        "publish_date": _publish_date(clean_text),
        "issuing_agencies": _issuing_agencies(clean_text),
        "policy_level": None,
        "policy_type": None,
        "policy_status": None,
        "source_url": source_url,
    }
    return {
        "input_type": input_type,
        "raw_input": raw_input,
        "content_hash": digest,
        "policy_id": policy_id,
        "title": title,
        "source_url": source_url,
        "text": clean_text,
        "metadata": metadata,
        "chunks": chunks,
        "search_text": _clip_for_search(f"{title}\n{clean_text}"),
    }


def validate_policy_text(text: str, title: str | None = None, source_url: str | None = None) -> None:
    """Fail fast when a fetched URL did not yield usable policy text."""

    clean_text = _compact_text(text)
    source_label = f"：{source_url}" if source_url else ""
    if len(clean_text) < MIN_EXTRACTED_POLICY_CHARS:
        raise SourcePolicyError(f"正文质量校验失败，提取到的政策正文过短{source_label}。请粘贴完整政策正文。")

    first_screen = clean_text[:800].lower()
    if any(marker in first_screen for marker in ERROR_PAGE_MARKERS):
        raise SourcePolicyError(f"正文质量校验失败，链接返回错误页、登录页或访问受限页面{source_label}。请粘贴完整政策正文。")

    marker_count = _policy_marker_count(f"{title or ''}\n{clean_text}")
    if marker_count < 2:
        raise SourcePolicyError(f"正文质量校验失败，链接内容不像正式政策正文{source_label}。请粘贴完整政策正文。")


def build_source_policy_from_local_policy(content: dict[str, Any], raw_input: str) -> dict[str, Any]:
    metadata = dict(content.get("metadata") or {})
    chunks = list(content.get("chunks") or [])
    text = "\n".join(str(chunk.get("content") or "") for chunk in chunks).strip()
    if not metadata or not chunks or not text:
        raise SourcePolicyError("Local policy content is incomplete")
    title = str(metadata.get("title") or metadata.get("policy_id") or "")
    return {
        "input_type": "legacy_policy_lookup",
        "raw_input": raw_input,
        "content_hash": sha256(text.encode("utf-8")).hexdigest(),
        "policy_id": str(metadata.get("policy_id") or ""),
        "title": title,
        "source_url": metadata.get("source_url"),
        "text": text,
        "metadata": metadata,
        "chunks": chunks,
        "search_text": _clip_for_search(f"{title}\n{text}"),
    }


def _is_pdf_url(url: str, content_type: str) -> bool:
    return "application/pdf" in content_type.lower() or urlparse(url).path.lower().endswith(".pdf")


def _chunk_source_policy(policy_id: str, text: str, max_chars: int = 1200) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    search_from = 0
    for section in split_sections(text):
        units = split_policy_units(section.content)
        grouped_units = group_units(units, max_chars=max_chars)
        for chunk_index, content in enumerate(grouped_units, start=1):
            char_start = text.find(content, search_from)
            if char_start == -1:
                char_start = text.find(content)
            if char_start == -1:
                char_start = search_from
            char_end = char_start + len(content)
            search_from = char_end
            chunks.append(
                {
                    "chunk_id": f"{policy_id}-S{section.index:03d}-C{chunk_index:03d}",
                    "policy_id": policy_id,
                    "section_title": section.title,
                    "section_index": section.index,
                    "chunk_index": chunk_index,
                    "page_start": None,
                    "page_end": None,
                    "char_start": char_start,
                    "char_end": char_end,
                    "previous_chunk_id": None,
                    "next_chunk_id": None,
                    "content": content,
                }
            )
    for index, chunk in enumerate(chunks):
        chunk["previous_chunk_id"] = chunks[index - 1]["chunk_id"] if index > 0 else None
        chunk["next_chunk_id"] = chunks[index + 1]["chunk_id"] if index < len(chunks) - 1 else None
    return chunks


def _pdf_text(payload: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(payload))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise SourcePolicyError(f"Failed to extract PDF policy text: {exc}") from exc


def _decode_bytes(payload: bytes, content_type: str) -> str:
    charset_match = re.search(r"charset=([\w.-]+)", content_type, flags=re.I)
    encodings = [charset_match.group(1)] if charset_match else []
    encodings.extend(["utf-8", "gb18030"])
    for encoding in encodings:
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def _html_text(html: str) -> tuple[str, str]:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    title = _collapse_whitespace(unescape(title_match.group(1))) if title_match else ""
    candidates = _html_main_text_candidates(html)
    parser = _HTMLTextExtractor()
    parser.feed(html)
    fallback = parser.text()
    candidate_text = _best_html_text(candidates)
    if candidate_text and _policy_marker_count(candidate_text) >= 2 and len(candidate_text) >= MIN_EXTRACTED_POLICY_CHARS:
        text = candidate_text
    else:
        text = fallback
    return title, text


def _html_main_text_candidates(html: str) -> list[str]:
    candidates: list[str] = []
    block_patterns = (
        r"<article\b[^>]*>(.*?)</article>",
        r"<main\b[^>]*>(.*?)</main>",
        r"<div\b[^>]*(?:article|content|main|TRS_Editor|Custom_UnionStyle|正文)[^>]*>(.*?)</div>",
    )
    for pattern in block_patterns:
        for match in re.finditer(pattern, html, flags=re.I | re.S):
            candidates.append(_strip_html(match.group(1)))
    return [candidate for candidate in candidates if candidate]


def _strip_html(fragment: str) -> str:
    fragment = re.sub(r"<(script|style|noscript|svg)\b[^>]*>.*?</\1>", " ", fragment, flags=re.I | re.S)
    fragment = re.sub(r"</(?:p|div|li|tr|h1|h2|h3|br)>", "\n", fragment, flags=re.I)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    return _compact_text(unescape(fragment))


def _best_html_text(candidates: list[str]) -> str:
    scored = []
    for candidate in candidates:
        scored.append((_policy_marker_count(candidate), len(candidate), candidate))
    if not scored:
        return ""
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag.lower() in {"p", "div", "br", "li", "tr", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag.lower() in {"p", "div", "li", "tr", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return _compact_text(unescape("".join(self._parts)))


def _compact_text(text: str) -> str:
    lines = [_collapse_whitespace(line) for line in text.replace("\r", "\n").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _title_from_text(text: str) -> str:
    quoted_title = re.search(r"《([^》]{4,100}(?:办法|通知|意见|规定|条例|规划|方案|细则|措施))》", text[:2000])
    if quoted_title:
        return quoted_title.group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:60]:
        if _is_article_line(line) or _is_agency_line(line) or _is_doc_number_line(line) or _is_date_line(line):
            continue
        if 4 <= len(line) <= 100 and _policy_title_marker_count(line) >= 1:
            return line

    for line in lines:
        stripped = line.strip()
        if 4 <= len(stripped) <= 120 and not _is_article_line(stripped):
            return stripped
    return ""


def _best_title(html_title: str | None, text: str) -> str:
    text_title = _title_from_text(text)
    if text_title and _policy_marker_count(text_title) >= 1:
        return text_title
    return (html_title or text_title or "").strip()


def _policy_marker_count(text: str) -> int:
    return sum(1 for marker in POLICY_TEXT_MARKERS if marker in text)


def _policy_title_marker_count(text: str) -> int:
    return sum(1 for marker in ("办法", "通知", "意见", "规定", "条例", "规划", "方案", "细则", "措施") if marker in text)


def _title_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/").split("/")[-1]
    return unescape(path) or url


def _document_number(text: str) -> str | None:
    match = re.search(r"[\u4e00-\u9fff]{1,12}\u3014?\d{4}\u3015?\s*\d+\u53f7", text)
    if match:
        return match.group(0)
    lines = [line.strip() for line in text.splitlines()[:20] if line.strip()]
    for index, line in enumerate(lines):
        if re.fullmatch(r"\u7b2c\s*\d+\s*\u53f7", line):
            if index > 0 and lines[index - 1] == "\u4ee4":
                return f"\u4ee4{line}"
            return line
    return None


def _publish_date(text: str) -> str | None:
    prefix = text[: text.find("\u7b2c\u4e00\u6761")] if "\u7b2c\u4e00\u6761" in text else text[:2000]
    matches = list(re.finditer(r"(20\d{2}|19\d{2})[.\-/\u5e74](\d{1,2})[.\-/\u6708](\d{1,2})\u65e5?", prefix))
    if not matches:
        return None
    date_line_matches = [match for match in matches if _is_date_line(_line_containing(prefix, match.start()))]
    match = date_line_matches[-1] if date_line_matches else matches[-1]
    year, month, day = match.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _issuing_agencies(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()[:12] if line.strip()]
    agencies: list[str] = []
    for line in lines:
        if _is_agency_line(line):
            agencies.append(line)
    return agencies[:3]


def _is_agency_line(line: str) -> bool:
    if _is_article_line(line) or len(line) > 80:
        return False
    if any(marker in line for marker in ("应当", "制定本", "服务提供者", "主管部门应当", "第一条", "第二条")):
        return False
    return bool(
        re.search(
            r"(国务院|办公厅|办公室|国家.+局|国家.+委|中华人民共和国.+部|.+委员会|.+人民政府|.+厅|.+局|.+部)$",
            line,
        )
    )


def _is_article_line(line: str) -> bool:
    return bool(re.match(r"^第[一二三四五六七八九十百千万\d]+条", line.strip()))


def _is_doc_number_line(line: str) -> bool:
    return line in {"令"} or bool(re.fullmatch(r"第\s*\d+\s*号", line.strip()))


def _is_date_line(line: str) -> bool:
    return bool(re.fullmatch(r"\s*(20\d{2}|19\d{2})[.\-/年]\d{1,2}[.\-/月]\d{1,2}日?\s*", line.strip()))


def _line_containing(text: str, offset: int) -> str:
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def _clip_for_search(text: str, max_chars: int = 1800) -> str:
    compacted = _collapse_whitespace(text)
    return compacted[:max_chars]
