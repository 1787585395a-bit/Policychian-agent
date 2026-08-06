from __future__ import annotations

import re
from typing import Any


PROHIBITED_INVESTMENT_TERMS = (
    "涔板叆",
    "鍗栧嚭",
    "鐩爣浠?",
    "鎺ㄨ崘鑲＄エ",
    "纭畾鎬ф敹鐩?",
    "纭畾鎬ф姇璧勫缓璁?",
    "买入",
    "卖出",
    "目标价",
    "推荐股票",
    "确定性收益",
    "确定性投资建议",
    "对于投资者而言",
    "投资者应重点关注",
    "投资者可重点关注",
    "应重点关注",
    "确定性趋势",
    "确定性需求",
    "利好",
    "利空",
    "成长叙事",
)

STRICT_SAFETY_PROFILE = "strict"
REPORT_WRITER_SAFETY_PROFILE = "report_writer"
REPORT_WRITER_ALLOWED_TERMS = (
    "利好",
    "利空",
    "应重点关注",
    "确定性趋势",
    "确定性需求",
    "成长叙事",
)


class SafetyViolation(ValueError):
    """Raised when generated output violates PolicyChain safety boundaries."""


def contains_prohibited_terms(payload: Any, *, profile: str = STRICT_SAFETY_PROFILE) -> list[str]:
    """Return prohibited investment-advice terms found in a generated payload."""

    text = _compact_safety_text(payload)
    prohibited_terms = _prohibited_terms_for_profile(profile)
    return [term for term in prohibited_terms if _compact_safety_text(term) in text]


def assert_no_investment_advice(
    payload: Any,
    context: str = "output",
    *,
    profile: str = STRICT_SAFETY_PROFILE,
) -> None:
    """Reject generated content that crosses the investment-advice boundary."""

    terms = contains_prohibited_terms(payload, profile=profile)
    if terms:
        joined = ", ".join(terms)
        raise SafetyViolation(f"{context} contains prohibited investment term(s): {joined}")


def _prohibited_terms_for_profile(profile: str) -> tuple[str, ...]:
    if profile == STRICT_SAFETY_PROFILE:
        return PROHIBITED_INVESTMENT_TERMS
    if profile == REPORT_WRITER_SAFETY_PROFILE:
        allowed = set(REPORT_WRITER_ALLOWED_TERMS)
        return tuple(term for term in PROHIBITED_INVESTMENT_TERMS if term not in allowed)
    raise ValueError(f"Unknown safety profile: {profile}")


def _compact_safety_text(payload: Any) -> str:
    return re.sub(r"[\W_]+", "", str(payload), flags=re.UNICODE).lower()
