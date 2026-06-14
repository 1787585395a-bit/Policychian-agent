from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from policychain.ingestion.id_generator import generate_policy_id
from policychain.ingestion.loaders import LoadedPolicyFile
from policychain.ingestion.normalizer import safe_filename_part
from policychain.schemas.policy_schema import PolicyMetadata


@dataclass
class ManifestRecord:
    sequence: int
    title: str
    issuing_agency: str
    document_number: str | None
    publish_date: str | None
    policy_status: str | None
    policy_type: str | None
    source_url: str | None
    local_filename: str


AGENCY_CODE_RULES: tuple[tuple[str, str], ...] = (
    ("国务院", "SC"),
    ("国家发展改革委", "NDRC"),
    ("发展改革委", "NDRC"),
    ("工业和信息化部", "MIIT"),
    ("科技部", "MOST"),
    ("教育部", "MOE"),
    ("交通运输部", "MOT"),
    ("自然资源部", "MNR"),
    ("国家药监局", "NMPA"),
    ("国家知识产权局", "CNIPA"),
    ("应急管理部", "MEM"),
    ("国家级", "NAT"),
)

REGION_CODE_RULES: dict[str, str] = {
    "国家": "",
    "广东": "GD",
    "山东": "SD",
    "北京": "BJ",
    "上海": "SH",
    "深圳": "SZ",
}


def extract_metadata(
    loaded_file: LoadedPolicyFile,
    file_hash: str,
    manifest_path: str | Path | None = None,
    policy_id: str | None = None,
) -> PolicyMetadata:
    record = find_manifest_record(manifest_path, loaded_file.original_filename) if manifest_path else None
    filename_parts = parse_policy_filename(loaded_file.original_filename)

    title = record.title if record else filename_parts.get("title") or infer_title_from_text(loaded_file.text)
    publish_date = record.publish_date if record else filename_parts.get("publish_date")
    year = (publish_date or filename_parts.get("year") or "0000")[:4]
    sequence = record.sequence if record else int(filename_parts.get("sequence") or 1)
    agency = record.issuing_agency if record else filename_parts.get("scope") or ""
    agency_code = agency_to_code(agency)
    region_code = region_to_code(filename_parts.get("scope"))

    generated_policy_id = policy_id or generate_policy_id(
        year=year,
        agency_code=agency_code,
        sequence=sequence,
        region_code=region_code or None,
    )

    policy_level = infer_policy_level(filename_parts.get("scope"), agency)
    normalized_filename = build_normalized_filename(
        publish_date=publish_date,
        agency=agency,
        title=title,
        policy_id=generated_policy_id,
        file_type=loaded_file.file_type,
    )

    return PolicyMetadata(
        policy_id=generated_policy_id,
        title=title,
        document_number=record.document_number if record else None,
        publish_date=publish_date,
        issuing_agencies=[agency] if agency else [],
        policy_level=policy_level,
        policy_type=record.policy_type if record else None,
        geographic_scope=filename_parts.get("scope"),
        policy_status=record.policy_status if record else None,
        source_url=record.source_url if record else None,
        original_filename=loaded_file.original_filename,
        normalized_filename=normalized_filename,
        file_hash=file_hash,
        file_type=loaded_file.file_type,
    )


def find_manifest_record(manifest_path: str | Path | None, filename: str) -> ManifestRecord | None:
    if manifest_path is None:
        return None

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Policy manifest does not exist: {path}")

    for row in _read_manifest_rows(path):
        if row.get("本地文件名") == filename:
            return ManifestRecord(
                sequence=int(row["序号"]),
                title=row.get("政策名称", "").strip(),
                issuing_agency=row.get("发文机关", "").strip(),
                document_number=_none_if_empty(row.get("发文字号")),
                publish_date=_none_if_empty(row.get("发布日期")),
                policy_status=_none_if_empty(row.get("有效状态")),
                policy_type=_none_if_empty(row.get("主题分类")),
                source_url=_none_if_empty(row.get("官方来源网址")),
                local_filename=filename,
            )
    return None


def parse_policy_filename(filename: str) -> dict[str, str]:
    stem = Path(filename).stem
    match = re.match(r"^(?P<sequence>\d+)_+(?P<scope>[^_]+)_+(?P<year>\d{4})_+(?P<title>.+)$", stem)
    if not match:
        return {"title": stem}
    parts = match.groupdict()
    parts["publish_date"] = parts["year"]
    return parts


def infer_title_from_text(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) <= 120:
            return stripped
    return "未命名政策文件"


def agency_to_code(agency: str | None) -> str:
    value = agency or ""
    for marker, code in AGENCY_CODE_RULES:
        if marker in value:
            return code
    return "NAT"


def region_to_code(scope: str | None) -> str:
    if not scope:
        return ""
    return REGION_CODE_RULES.get(scope, "")


def infer_policy_level(scope: str | None, agency: str | None) -> str:
    if scope == "国家" or (agency and "国家级" in agency):
        return "national"
    if scope:
        return "local"
    return "unknown"


def build_normalized_filename(
    publish_date: str | None,
    agency: str,
    title: str,
    policy_id: str,
    file_type: str,
) -> str:
    date_part = (publish_date or "unknown").replace("-", "")
    agency_part = safe_filename_part(agency or "unknown_agency", max_length=30)
    title_part = safe_filename_part(title, max_length=60)
    return f"{date_part}_{agency_part}_{title_part}_{policy_id}.{file_type}"


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as source:
                return list(csv.DictReader(source))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("manifest", b"", 0, 1, f"Unable to decode manifest: {path}")


def _none_if_empty(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
