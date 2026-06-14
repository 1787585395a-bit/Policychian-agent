from __future__ import annotations

import unittest

from policychain.agents import (
    match_companies_for_impacts,
    run_company_matcher,
    run_impact_analyst,
    run_policy_analyst,
)
from policychain.schemas import CompanyMatchOutput
from policychain.state import PolicyResearchState
from tests.helpers import build_sample_store


PROHIBITED_TERMS = ("买入", "卖出", "目标价", "推荐股票")
MATCH_LEVELS = {"high", "medium", "low"}


class CompanyMatcherTests(unittest.TestCase):
    def test_match_companies_for_impacts_returns_structured_matches(self) -> None:
        impacts = [
            {
                "industry": "生成式人工智能服务",
                "impact_type": "direct",
                "direction": "mixed",
                "transmission_logic": "政策直接规范生成式人工智能服务提供、备案和投诉受理责任。",
                "conditions": ["需要结合监管执行口径判断。"],
                "risks": ["合规成本可能上升。"],
            }
        ]
        output = match_companies_for_impacts(impacts)
        result = output.to_dict()

        self.assertIsInstance(output, CompanyMatchOutput)
        self.assertGreaterEqual(len(result["companies"]), 1)
        company = result["companies"][0]
        self.assertEqual(company["industry_segment"], "生成式人工智能服务")
        self.assertIn(company["match_level"], MATCH_LEVELS)
        self.assertTrue(company["business_evidence"])
        self.assertTrue(company["data_date"])
        self.assertGreater(company["confidence"], 0)
        self.assertTrue(result["uncertainties"])

    def test_run_company_matcher_writes_shared_state(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
            run_policy_analyst(state, store)
            run_impact_analyst(state)
            output = run_company_matcher(state)

            self.assertGreaterEqual(len(output.companies), 1)
            self.assertEqual(len(state.company_candidates), len(state.company_matches))
            self.assertGreaterEqual(len(state.company_matches), 1)
            self.assertTrue(state.uncertainties)
        finally:
            store.close()

    def test_company_matches_include_business_evidence_and_policy_link(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
            run_policy_analyst(state, store)
            run_impact_analyst(state)
            output = run_company_matcher(state)
            companies = output.to_dict()["companies"]

            self.assertTrue(any(company["policy_link"] for company in companies))
            self.assertTrue(all(company["business_evidence"] for company in companies))
            self.assertTrue(all(company["revenue_relevance"] for company in companies))
            self.assertTrue(all(0.0 <= company["confidence"] <= 1.0 for company in companies))
        finally:
            store.close()

    def test_company_matcher_missing_impacts_returns_uncertainty(self) -> None:
        state = PolicyResearchState(user_query="生成式人工智能")
        output = run_company_matcher(state)
        result = output.to_dict()

        self.assertEqual(result["companies"], [])
        self.assertTrue(result["uncertainties"])
        self.assertEqual(state.company_matches, [])

    def test_company_matcher_does_not_emit_investment_advice_terms(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="生成式人工智能 公司影响")
            run_policy_analyst(state, store)
            run_impact_analyst(state)
            output = run_company_matcher(state)
            rendered = str(output.to_dict())

            for term in PROHIBITED_TERMS:
                self.assertNotIn(term, rendered)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
