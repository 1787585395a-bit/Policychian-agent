from __future__ import annotations

from typing import Any


PROHIBITED_INVESTMENT_TERMS = (
    "买入",
    "卖出",
    "目标价",
    "推荐股票",
    "确定性收益",
    "确定性投资建议",
)


class SafetyViolation(ValueError):
    """Raised when generated output violates PolicyChain safety boundaries."""


def contains_prohibited_terms(payload: Any) -> list[str]:
    """Return prohibited investment-advice terms found in a generated payload."""

    text = str(payload)
    return [term for term in PROHIBITED_INVESTMENT_TERMS if term in text]


def assert_no_investment_advice(payload: Any, context: str = "output") -> None:
    """Reject generated content that crosses the investment-advice boundary."""

    terms = contains_prohibited_terms(payload)
    if terms:
        joined = ", ".join(terms)
        raise SafetyViolation(f"{context} contains prohibited investment term(s): {joined}")
