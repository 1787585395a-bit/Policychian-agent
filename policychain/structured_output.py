from __future__ import annotations

import json
import re
from typing import Any, Callable, TypeVar

from policychain.safety import assert_no_investment_advice
from policychain.observability import record_event
from policychain.schemas.agent_outputs import (
    CompanyEvidence,
    CompanyMatch,
    CompanyMatchOutput,
    EvidenceItem,
    ImpactAnalysisOutput,
    ImplementationStep,
    IndustryImpact,
    PolicyAnalysisOutput,
    StrengthAssessment,
)


T = TypeVar("T")


class StructuredOutputError(ValueError):
    """Raised when an LLM response cannot be parsed into a supported schema."""


SCHEMA_BUILDERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "PolicyAnalysisOutput": lambda payload: _build_policy_analysis(payload),
    "ImpactAnalysisOutput": lambda payload: _build_impact_analysis(payload),
    "CompanyMatchOutput": lambda payload: _build_company_match_output(payload),
}


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw model text, including fenced JSON responses."""

    if not text.strip():
        raise StructuredOutputError("Structured output is empty")

    candidate = _extract_fenced_json(text) or text.strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        payload = _parse_embedded_json_object(text)

    if not isinstance(payload, dict):
        raise StructuredOutputError("Structured output must be a JSON object")
    return payload


def parse_structured_output(text: str, schema_name: str) -> Any:
    """Parse and validate an LLM response against a supported output schema."""

    try:
        assert_no_investment_advice(text, context=f"{schema_name} raw output")
        payload = parse_json_object(text)
        output = validate_structured_payload(payload, schema_name)
        assert_no_investment_advice(output.to_dict(), context=schema_name)
    except Exception as exc:
        record_event(
            "schema.validation",
            stage=schema_name,
            status="error",
            schema=schema_name,
            error=f"{exc.__class__.__name__}: {str(exc)[:300]}",
        )
        raise
    record_event("schema.validation", stage=schema_name, status="ok", schema=schema_name)
    return output


def validate_structured_payload(payload: dict[str, Any], schema_name: str) -> Any:
    try:
        builder = SCHEMA_BUILDERS[schema_name]
    except KeyError as exc:
        available = ", ".join(sorted(SCHEMA_BUILDERS))
        raise StructuredOutputError(f"Unsupported structured output schema: {schema_name}. Available: {available}") from exc
    return builder(payload)


def _build_policy_analysis(payload: dict[str, Any]) -> PolicyAnalysisOutput:
    _require_fields(
        payload,
        "PolicyAnalysisOutput",
        (
            "policy_identity",
            "policy_goals",
            "target_entities",
            "policy_measures",
            "historical_changes",
            "strength_assessment",
            "evidence",
            "uncertainties",
        ),
    )
    return PolicyAnalysisOutput(
        policy_identity=_require_dict(payload, "policy_identity"),
        policy_goals=_require_list_of_str(payload, "policy_goals"),
        target_entities=_require_list_of_str(payload, "target_entities"),
        policy_measures=_require_list_of_str(payload, "policy_measures"),
        historical_changes=_coerce_historical_changes(payload),
        strength_assessment=_build_strength_assessment(_require_dict(payload, "strength_assessment")),
        evidence=_build_evidence_list(_require_list(payload, "evidence"), "evidence"),
        uncertainties=_coerce_explanatory_list(payload, "uncertainties"),
    )


def _build_impact_analysis(payload: dict[str, Any]) -> ImpactAnalysisOutput:
    _require_fields(
        payload,
        "ImpactAnalysisOutput",
        (
            "implementation_actors",
            "implementation_mechanisms",
            "implementation_chain",
            "industry_impacts",
            "uncertainties",
            "evidence",
        ),
    )
    return ImpactAnalysisOutput(
        implementation_actors=_require_list_of_str(payload, "implementation_actors"),
        implementation_mechanisms=_require_list_of_str(payload, "implementation_mechanisms"),
        implementation_chain=[
            _build_implementation_step(item, f"implementation_chain[{index}]")
            for index, item in enumerate(_require_list(payload, "implementation_chain"))
        ],
        industry_impacts=[
            _build_industry_impact(item, f"industry_impacts[{index}]")
            for index, item in enumerate(_require_list(payload, "industry_impacts"))
        ],
        uncertainties=_coerce_explanatory_list(payload, "uncertainties"),
        evidence=_build_evidence_list(_require_list(payload, "evidence"), "evidence"),
    )


def _build_company_match_output(payload: dict[str, Any]) -> CompanyMatchOutput:
    _require_fields(payload, "CompanyMatchOutput", ("companies", "uncertainties"))
    return CompanyMatchOutput(
        companies=[
            _build_company_match(item, f"companies[{index}]")
            for index, item in enumerate(_require_list(payload, "companies"))
        ],
        uncertainties=_coerce_explanatory_list(payload, "uncertainties"),
    )


def _build_strength_assessment(payload: dict[str, Any]) -> StrengthAssessment:
    _require_fields(payload, "strength_assessment", ("level", "reasons", "uncertainties"))
    level = _require_str(payload, "level")
    if level not in {"high", "medium", "low", "unknown"}:
        raise StructuredOutputError("strength_assessment.level must be high, medium, low, or unknown")
    return StrengthAssessment(
        level=level,
        reasons=_require_list_of_str(payload, "reasons"),
        uncertainties=_coerce_explanatory_list(payload, "uncertainties"),
    )


def _build_evidence_list(items: list[Any], field_name: str) -> list[EvidenceItem]:
    return [_build_evidence_item(item, f"{field_name}[{index}]") for index, item in enumerate(items)]


def _build_evidence_item(payload: Any, field_name: str) -> EvidenceItem:
    if not isinstance(payload, dict):
        raise StructuredOutputError(f"{field_name} must be an object")
    _require_fields(payload, field_name, ("policy_id", "chunk_id", "source_url", "text", "note"))
    return EvidenceItem(
        policy_id=_require_str(payload, "policy_id", field_name),
        chunk_id=_optional_str(payload, "chunk_id", field_name),
        source_url=_optional_str(payload, "source_url", field_name),
        text=_require_str(payload, "text", field_name),
        note=_optional_str(payload, "note", field_name),
    )


def _build_implementation_step(payload: Any, field_name: str) -> ImplementationStep:
    if not isinstance(payload, dict):
        raise StructuredOutputError(f"{field_name} must be an object")
    _require_fields(payload, field_name, ("step_index", "actor", "action", "mechanism", "evidence"))
    step_index = payload["step_index"]
    if not isinstance(step_index, int) or isinstance(step_index, bool) or step_index <= 0:
        raise StructuredOutputError(f"{field_name}.step_index must be a positive integer")
    return ImplementationStep(
        step_index=step_index,
        actor=_require_str(payload, "actor", field_name),
        action=_require_str(payload, "action", field_name),
        mechanism=_require_str(payload, "mechanism", field_name),
        evidence=_build_evidence_list(_require_list(payload, "evidence", field_name), f"{field_name}.evidence"),
    )


def _build_industry_impact(payload: Any, field_name: str) -> IndustryImpact:
    if not isinstance(payload, dict):
        raise StructuredOutputError(f"{field_name} must be an object")
    _require_fields(
        payload,
        field_name,
        ("industry", "impact_type", "direction", "transmission_logic", "conditions", "risks", "evidence"),
    )
    impact_type = _require_str(payload, "impact_type", field_name)
    if impact_type not in {"direct", "indirect", "potential"}:
        raise StructuredOutputError(f"{field_name}.impact_type must be direct, indirect, or potential")
    direction = _require_str(payload, "direction", field_name)
    if direction not in {"positive", "negative", "mixed", "neutral", "unknown"}:
        raise StructuredOutputError(f"{field_name}.direction must be positive, negative, mixed, neutral, or unknown")
    return IndustryImpact(
        industry=_require_str(payload, "industry", field_name),
        impact_type=impact_type,
        direction=direction,
        transmission_logic=_require_str(payload, "transmission_logic", field_name),
        policy_measure=_optional_str(payload, "policy_measure", field_name) or "",
        implementation_action=_optional_str(payload, "implementation_action", field_name) or "",
        chain_segment=_optional_str(payload, "chain_segment", field_name) or "",
        business_variables=_optional_list_of_str(payload, "business_variables", field_name),
        affected_company_types=_optional_list_of_str(payload, "affected_company_types", field_name),
        conditions=_require_list_of_str(payload, "conditions", field_name),
        risks=_require_list_of_str(payload, "risks", field_name),
        evidence=_build_evidence_list(_require_list(payload, "evidence", field_name), f"{field_name}.evidence"),
    )


def _build_company_match(payload: Any, field_name: str) -> CompanyMatch:
    if not isinstance(payload, dict):
        raise StructuredOutputError(f"{field_name} must be an object")
    _require_fields(
        payload,
        field_name,
        (
            "company_name",
            "industry_segment",
            "matched_business",
            "match_level",
            "business_evidence",
            "policy_link",
            "revenue_relevance",
            "conditions",
            "risks",
            "data_date",
            "confidence",
        ),
    )
    match_level = _require_str(payload, "match_level", field_name)
    if match_level not in {"high", "medium", "low"}:
        raise StructuredOutputError(f"{field_name}.match_level must be high, medium, or low")
    confidence = payload["confidence"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        raise StructuredOutputError(f"{field_name}.confidence must be a number between 0 and 1")
    return CompanyMatch(
        company_name=_require_str(payload, "company_name", field_name),
        industry_segment=_require_str(payload, "industry_segment", field_name),
        matched_business=_require_str(payload, "matched_business", field_name),
        match_level=match_level,
        impact_id=_optional_str(payload, "impact_id", field_name) or "",
        impact_industry=_optional_str(payload, "impact_industry", field_name) or "",
        stock_code=_optional_str(payload, "stock_code", field_name) or "",
        chain_segment=_optional_str(payload, "chain_segment", field_name) or "",
        related_product_or_business=_optional_str(payload, "related_product_or_business", field_name) or "",
        revenue_or_ratio=_optional_str(payload, "revenue_or_ratio", field_name) or "",
        source_url=_optional_str(payload, "source_url", field_name),
        match_conditions=_optional_list_of_str(payload, "match_conditions", field_name),
        negative_evidence=_optional_list_of_str(payload, "negative_evidence", field_name),
        business_evidence=[
            _build_company_evidence(item, f"{field_name}.business_evidence[{index}]")
            for index, item in enumerate(_require_list(payload, "business_evidence", field_name))
        ],
        policy_link=_require_str(payload, "policy_link", field_name),
        revenue_relevance=_require_str(payload, "revenue_relevance", field_name),
        conditions=_require_list_of_str(payload, "conditions", field_name),
        risks=_require_list_of_str(payload, "risks", field_name),
        data_date=_require_str(payload, "data_date", field_name),
        confidence=float(confidence),
        audit_status=_optional_str(payload, "audit_status", field_name) or "pending",
        audit_reason=_optional_str(payload, "audit_reason", field_name) or "",
    )


def _build_company_evidence(payload: Any, field_name: str) -> CompanyEvidence:
    if not isinstance(payload, dict):
        raise StructuredOutputError(f"{field_name} must be an object")
    _require_fields(payload, field_name, ("source_name", "source_url", "text", "data_date"))
    return CompanyEvidence(
        source_name=_require_str(payload, "source_name", field_name),
        source_url=_optional_str(payload, "source_url", field_name),
        text=_require_str(payload, "text", field_name),
        data_date=_optional_str(payload, "data_date", field_name) or "unknown",
        revenue_or_ratio=_optional_str(payload, "revenue_or_ratio", field_name) or "",
    )


def _require_fields(payload: dict[str, Any], object_name: str, fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise StructuredOutputError(f"{object_name} missing required field(s): {', '.join(missing)}")


def _require_dict(payload: dict[str, Any], field_name: str, parent: str | None = None) -> dict[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise StructuredOutputError(f"{_field_path(field_name, parent)} must be an object")
    return value


def _require_list(payload: dict[str, Any], field_name: str, parent: str | None = None) -> list[Any]:
    value = payload.get(field_name)
    if not isinstance(value, list):
        raise StructuredOutputError(f"{_field_path(field_name, parent)} must be a list")
    return value


def _require_list_of_str(payload: dict[str, Any], field_name: str, parent: str | None = None) -> list[str]:
    values = _require_list(payload, field_name, parent)
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise StructuredOutputError(f"{_field_path(field_name, parent)}[{index}] must be a string")
    return list(values)


def _coerce_historical_changes(payload: dict[str, Any]) -> list[str]:
    values = _require_list(payload, "historical_changes")
    changes: list[str] = []
    for index, value in enumerate(values):
        if isinstance(value, str):
            if value.strip():
                changes.append(value)
            continue
        if isinstance(value, dict):
            text = _compact_history_change(value)
            if text:
                changes.append(text)
            continue
        raise StructuredOutputError(f"historical_changes[{index}] must be a string or object")
    return changes


def _compact_history_change(value: dict[str, Any]) -> str:
    preferred_keys = (
        "policy",
        "policy_title",
        "title",
        "date",
        "change",
        "difference",
        "summary",
        "evidence",
        "source",
    )
    parts: list[str] = []
    for key in preferred_keys:
        if key not in value:
            continue
        item = value.get(key)
        if isinstance(item, (str, int, float)) and str(item).strip():
            parts.append(f"{key}: {item}")
        elif isinstance(item, list):
            joined = "；".join(str(part).strip() for part in item if str(part).strip())
            if joined:
                parts.append(f"{key}: {joined}")
    if not parts:
        parts = [f"{key}: {item}" for key, item in value.items() if isinstance(item, (str, int, float)) and str(item).strip()]
    return "；".join(parts)


def _coerce_explanatory_list(payload: dict[str, Any], field_name: str, parent: str | None = None) -> list[str]:
    values = _require_list(payload, field_name, parent)
    items: list[str] = []
    for index, value in enumerate(values):
        text = _compact_explanatory_item(value)
        if text:
            items.append(text)
            continue
        if value is not None:
            raise StructuredOutputError(f"{_field_path(field_name, parent)}[{index}] must be a string-like explanation")
    return items


def _compact_explanatory_item(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_compact_explanatory_item(item) for item in value]
        return "; ".join(part for part in parts if part)
    if isinstance(value, dict):
        return _compact_explanatory_mapping(value)
    return ""


def _compact_explanatory_mapping(value: dict[str, Any]) -> str:
    preferred_keys = (
        "message",
        "uncertainty",
        "issue",
        "reason",
        "detail",
        "description",
        "evidence",
        "source",
        "tool",
        "error",
    )
    parts: list[str] = []
    for key in preferred_keys:
        if key not in value:
            continue
        text = _compact_explanatory_item(value.get(key))
        if text:
            parts.append(f"{key}: {text}")
    if not parts:
        for key, item in value.items():
            text = _compact_explanatory_item(item)
            if text:
                parts.append(f"{key}: {text}")
    return _clip_text("; ".join(parts), 600)


def _clip_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def _optional_list(payload: dict[str, Any], field_name: str, parent: str | None = None) -> list[Any]:
    if field_name not in payload or payload.get(field_name) is None:
        return []
    return _require_list(payload, field_name, parent)


def _optional_list_of_str(payload: dict[str, Any], field_name: str, parent: str | None = None) -> list[str]:
    if field_name not in payload or payload.get(field_name) is None:
        return []
    return _require_list_of_str(payload, field_name, parent)


def _require_str(payload: dict[str, Any], field_name: str, parent: str | None = None) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise StructuredOutputError(f"{_field_path(field_name, parent)} must be a non-empty string")
    return value


def _optional_str(payload: dict[str, Any], field_name: str, parent: str | None = None) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise StructuredOutputError(f"{_field_path(field_name, parent)} must be a string or null")
    return value


def _field_path(field_name: str, parent: str | None) -> str:
    if parent:
        return f"{parent}.{field_name}"
    return field_name


def _extract_fenced_json(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _parse_embedded_json_object(text: str) -> dict[str, Any]:
    candidate = _extract_first_balanced_object(text)
    if candidate is None:
        raise StructuredOutputError("Structured output is not valid JSON")
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError("Structured output contains malformed JSON object") from exc
    if not isinstance(payload, dict):
        raise StructuredOutputError("Structured output must be a JSON object")
    return payload


def _extract_first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
