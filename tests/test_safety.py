from __future__ import annotations

import unittest

from policychain.safety import (
    PROHIBITED_INVESTMENT_TERMS,
    REPORT_WRITER_ALLOWED_TERMS,
    REPORT_WRITER_SAFETY_PROFILE,
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

    def test_investor_directed_and_return_narrative_terms_are_rejected_with_spacing_or_punctuation(self) -> None:
        phrases = (
            "对于 投资者 而言",
            "投资者，应重点关注",
            "投资者 可 重点关注",
            "应、重点关注",
            "确定性，趋势",
            "确定性 需求",
            "构成利好",
            "存在利空",
            "成长-叙事",
        )

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertTrue(contains_prohibited_terms(phrase))
                with self.assertRaises(SafetyViolation):
                    assert_no_investment_advice(phrase)

    def test_neutral_company_watchlist_title_remains_allowed(self) -> None:
        assert_no_investment_advice("## A 股公司关注清单\n仅用于公司业务匹配研究。")

    def test_report_profile_accepts_conditional_operating_analysis_soft_terms(self) -> None:
        text = (
            "若项目落地，订单确定性需求可能形成阶段性利好；若成本传导不畅则可能形成利空。"
            "执行节奏应重点关注，当前确定性趋势和成长叙事仍取决于配套资金。"
        )

        self.assertEqual(
            contains_prohibited_terms(text, profile=REPORT_WRITER_SAFETY_PROFILE),
            [],
        )
        assert_no_investment_advice(text, profile=REPORT_WRITER_SAFETY_PROFILE)
        for term in REPORT_WRITER_ALLOWED_TERMS:
            self.assertIn(term, text)
            with self.assertRaises(SafetyViolation):
                assert_no_investment_advice(term)

    def test_report_profile_still_rejects_transactions_returns_and_investor_actions(self) -> None:
        phrases = (
            "买入",
            "卖出",
            "目标价",
            "推荐股票",
            "确定性收益",
            "确定性投资建议",
            "对于投资者而言",
            "投资者应重点关注",
            "投资者可重点关注",
        )

        for phrase in phrases:
            with self.subTest(phrase=phrase):
                with self.assertRaises(SafetyViolation):
                    assert_no_investment_advice(
                        phrase,
                        profile=REPORT_WRITER_SAFETY_PROFILE,
                    )


if __name__ == "__main__":
    unittest.main()
