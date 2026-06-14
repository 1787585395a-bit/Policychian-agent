from __future__ import annotations

import unittest

from policychain.safety import (
    PROHIBITED_INVESTMENT_TERMS,
    SafetyViolation,
    assert_no_investment_advice,
    contains_prohibited_terms,
)


class SafetyTests(unittest.TestCase):
    def test_contains_prohibited_terms_detects_investment_advice_terms(self) -> None:
        payload = {"analysis": "该公司不是推荐股票，也不应输出目标价。"}

        self.assertEqual(contains_prohibited_terms(payload), ["目标价", "推荐股票"])

    def test_assert_no_investment_advice_raises_with_context(self) -> None:
        with self.assertRaisesRegex(SafetyViolation, "Policy analysis output"):
            assert_no_investment_advice("输出包含买入结论", context="Policy analysis output")

    def test_disclaimer_without_prohibited_terms_is_allowed(self) -> None:
        assert_no_investment_advice("本报告仅用于政策研究和业务匹配分析，不构成任何投资建议。")

    def test_public_term_list_covers_project_boundary(self) -> None:
        for term in ("买入", "卖出", "目标价", "推荐股票", "确定性投资建议"):
            self.assertIn(term, PROHIBITED_INVESTMENT_TERMS)


if __name__ == "__main__":
    unittest.main()
