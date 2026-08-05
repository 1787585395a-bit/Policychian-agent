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


class SafetyViolation(ValueError):
    """Raised when generated output violates PolicyChain safety boundaries."""


def contains_prohibited_terms(payload: Any) -> list[str]:
    """Return prohibited investment-advice terms found in a generated payload."""

    text = _compact_safety_text(payload)
    return [term for term in PROHIBITED_INVESTMENT_TERMS if _compact_safety_text(term) in text]


def assert_no_investment_advice(payload: Any, context: str = "output") -> None:
    """Reject generated content that crosses the investment-advice boundary."""

    terms = contains_prohibited_terms(payload)
    if terms:
        joined = ", ".join(terms)
        raise SafetyViolation(f"{context} contains prohibited investment term(s): {joined}")


def _compact_safety_text(payload: Any) -> str:
    return re.sub(r"[\W_]+", "", str(payload), flags=re.UNICODE).lower()
