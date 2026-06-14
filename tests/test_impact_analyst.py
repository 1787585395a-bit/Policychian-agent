from __future__ import annotations

import unittest

from policychain.agents import analyze_policy_impact, run_impact_analyst, run_policy_analyst
from policychain.schemas import ImpactAnalysisOutput
from policychain.state import PolicyResearchState
from tests.helpers import build_sample_store


PROHIBITED_TERMS = ("买入", "卖出", "目标价", "推荐股票")
ALLOWED_IMPACT_TYPES = {"direct", "indirect", "potential"}
ALLOWED_DIRECTIONS = {"positive", "negative", "mixed"}


class ImpactAnalystTests(unittest.TestCase):
    def test_analyze_policy_impact_returns_structured_output(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
            run_policy_analyst(state, store)
            output = analyze_policy_impact(state.policy_analysis, state.policy_chunks)
            result = output.to_dict()

            self.assertIsInstance(output, ImpactAnalysisOutput)
            self.assertGreaterEqual(len(result["implementation_actors"]), 1)
            self.assertGreaterEqual(len(result["implementation_mechanisms"]), 1)
            self.assertGreaterEqual(len(result["implementation_chain"]), 1)
            self.assertGreaterEqual(len(result["industry_impacts"]), 1)
            self.assertLessEqual(len(result["industry_impacts"]), 5)
            self.assertTrue(result["uncertainties"])
        finally:
            store.close()

    def test_run_impact_analyst_writes_shared_state(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
            run_policy_analyst(state, store)
            output = run_impact_analyst(state)

            self.assertGreaterEqual(len(output.implementation_chain), 1)
            self.assertGreaterEqual(len(state.implementation_path), 1)
            self.assertGreaterEqual(len(state.industry_impacts), 1)
            self.assertGreaterEqual(len(state.evidence), 1)
            self.assertTrue(state.uncertainties)
        finally:
            store.close()

    def test_industry_impacts_have_bounded_types_logic_and_evidence(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
            run_policy_analyst(state, store)
            output = run_impact_analyst(state)
            impacts = output.to_dict()["industry_impacts"]

            self.assertTrue(any(impact["industry"] == "生成式人工智能服务" for impact in impacts))
            for impact in impacts:
                self.assertIn(impact["impact_type"], ALLOWED_IMPACT_TYPES)
                self.assertIn(impact["direction"], ALLOWED_DIRECTIONS)
                self.assertTrue(impact["transmission_logic"])
                self.assertTrue(impact["conditions"])
                self.assertTrue(impact["risks"])
                self.assertGreaterEqual(len(impact["evidence"]), 1)
        finally:
            store.close()

    def test_impact_analyst_missing_policy_analysis_returns_uncertainty(self) -> None:
        state = PolicyResearchState(user_query="生成式人工智能")
        output = run_impact_analyst(state)
        result = output.to_dict()

        self.assertEqual(result["implementation_chain"], [])
        self.assertEqual(result["industry_impacts"], [])
        self.assertTrue(result["uncertainties"])
        self.assertTrue(state.uncertainties)

    def test_impact_analyst_does_not_emit_investment_advice_terms(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="生成式人工智能 公司影响")
            run_policy_analyst(state, store)
            output = run_impact_analyst(state)
            rendered = str(output.to_dict())

            for term in PROHIBITED_TERMS:
                self.assertNotIn(term, rendered)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
