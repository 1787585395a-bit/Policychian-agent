from __future__ import annotations

import unittest

from policychain.graph import run_policy_research_workflow
from tests.helpers import build_sample_store


PROHIBITED_TERMS = ("买入", "卖出", "目标价", "推荐股票")


class EndToEndWorkflowTests(unittest.TestCase):
    def test_policy_research_workflow_runs_three_agents_and_report(self) -> None:
        store = build_sample_store()
        try:
            state = run_policy_research_workflow(
                "生成式人工智能服务提供者有哪些管理要求",
                store,
            )

            self.assertEqual(state.policy_ids, ["POL-2023-NAT-0048"])
            self.assertTrue(state.policy_analysis)
            self.assertGreaterEqual(len(state.implementation_path), 1)
            self.assertGreaterEqual(len(state.industry_impacts), 1)
            self.assertEqual(state.company_matches, [])
            self.assertIn("PolicyChain 政策研究报告", state.final_report)
            self.assertGreaterEqual(len(state.evidence), 1)
            self.assertTrue(state.uncertainties)
            self.assertTrue(any("候选公司" in item or "mock" in item for item in state.uncertainties))
        finally:
            store.close()

    def test_policy_research_workflow_report_has_no_prohibited_terms(self) -> None:
        store = build_sample_store()
        try:
            state = run_policy_research_workflow("生成式人工智能 公司影响", store)

            for term in PROHIBITED_TERMS:
                self.assertNotIn(term, state.final_report)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
