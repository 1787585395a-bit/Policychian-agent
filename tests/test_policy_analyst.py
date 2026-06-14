from __future__ import annotations

import unittest

from policychain.agents import analyze_policy_content, run_policy_analyst
from policychain.schemas import PolicyAnalysisOutput
from policychain.state import PolicyResearchState
from policychain.tools import read_policy_content
from tests.helpers import build_sample_store


PROHIBITED_TERMS = ("买入", "卖出", "目标价", "推荐股票")


class PolicyAnalystTests(unittest.TestCase):
    def test_analyze_policy_content_returns_structured_policy_analysis(self) -> None:
        store = build_sample_store()
        try:
            content = read_policy_content(store, "POL-2023-NAT-0048")
            output = analyze_policy_content(
                user_query="生成式人工智能 服务 管理",
                metadata=content["metadata"],
                chunks=content["chunks"],
            )

            self.assertIsInstance(output, PolicyAnalysisOutput)
            result = output.to_dict()
            self.assertEqual(result["policy_identity"]["policy_id"], "POL-2023-NAT-0048")
            self.assertEqual(result["policy_identity"]["title"], "生成式人工智能服务管理暂行办法")
            self.assertGreaterEqual(len(result["policy_goals"]), 1)
            self.assertGreaterEqual(len(result["policy_measures"]), 1)
            self.assertGreaterEqual(len(result["target_entities"]), 1)
            self.assertIn(result["strength_assessment"]["level"], {"low", "medium", "high", "unknown"})
            self.assertGreaterEqual(len(result["evidence"]), 1)
            self.assertTrue(result["uncertainties"])
        finally:
            store.close()

    def test_run_policy_analyst_writes_shared_state(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
            output = run_policy_analyst(state, store)

            self.assertEqual(output.policy_identity["policy_id"], "POL-2023-NAT-0048")
            self.assertEqual(state.policy_ids, ["POL-2023-NAT-0048"])
            self.assertEqual(state.policy_analysis["policy_identity"]["policy_id"], "POL-2023-NAT-0048")
            self.assertGreaterEqual(len(state.policy_chunks), 1)
            self.assertGreaterEqual(len(state.evidence), 1)
            self.assertTrue(state.uncertainties)
        finally:
            store.close()

    def test_policy_analyst_preserves_evidence_identity(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="生成式人工智能")
            output = run_policy_analyst(state, store)
            evidence = output.to_dict()["evidence"]

            self.assertGreaterEqual(len(evidence), 1)
            self.assertTrue(all(item["policy_id"] == "POL-2023-NAT-0048" for item in evidence))
            self.assertTrue(any(item["chunk_id"] for item in evidence))
            self.assertTrue(any(item["source_url"] for item in evidence))
        finally:
            store.close()

    def test_policy_analyst_no_result_returns_uncertainty(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="完全不存在的查询词")
            output = run_policy_analyst(state, store)
            result = output.to_dict()

            self.assertEqual(result["policy_identity"]["status"], "no_policy_found")
            self.assertEqual(result["evidence"], [])
            self.assertTrue(result["uncertainties"])
        finally:
            store.close()

    def test_policy_analyst_does_not_emit_investment_advice_terms(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="生成式人工智能 公司影响")
            output = run_policy_analyst(state, store)
            rendered = str(output.to_dict())

            for term in PROHIBITED_TERMS:
                self.assertNotIn(term, rendered)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
