from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from policychain.graph import run_llm_policy_research_workflow, run_policy_research_workflow
from policychain.mcp import FakeMCPInvoker
from tests.helpers import build_sample_store


PROHIBITED_TERMS = ("买入", "卖出", "目标价", "推荐股票")


class _NoopLLMClient:
    def generate(self, _system_prompt: str, _user_prompt: str) -> str:
        return "{}"


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
            self.assertTrue(any("Web-first" in item for item in state.uncertainties))
            self.assertTrue(all(item.get("retrieval_status") == "discovery_error" for item in state.company_coverage))
            self.assertIn("Web-first discovery 未执行", state.final_report)
            self.assertNotIn("查询成功但真实返回空", state.final_report)
        finally:
            store.close()

    def test_llm_graph_company_failure_respects_web_first_and_never_calls_legacy_recall(self) -> None:
        store = build_sample_store()
        invoker = FakeMCPInvoker({})

        def policy_stage(state, *_args, **_kwargs):
            state.policy_analysis = {"policy_identity": {"title": "测试政策"}}

        def impact_stage(state, *_args, **_kwargs):
            state.industry_impacts = [
                {
                    "impact_id": "IMP-001",
                    "industry": "反渗透膜",
                    "chain_segment": "反渗透膜",
                    "transmission_logic": "项目采购带动反渗透膜需求",
                    "business_variables": ["反渗透膜需求"],
                    "affected_company_types": ["反渗透膜供应商"],
                    "conditions": [],
                    "risks": [],
                }
            ]

        try:
            with (
                patch.dict(os.environ, {"POLICYCHAIN_COMPANY_DISCOVERY_MODE": "web_first"}),
                patch("policychain.graph.run_llm_policy_analyst", side_effect=policy_stage),
                patch("policychain.graph.run_llm_impact_analyst", side_effect=impact_stage),
                patch("policychain.graph.run_llm_company_matcher", side_effect=RuntimeError("discovery failed")),
                patch(
                    "policychain.graph.write_llm_research_report",
                    side_effect=lambda state, _client: setattr(state, "final_report", "report"),
                ),
            ):
                state = run_llm_policy_research_workflow(
                    "反渗透膜政策",
                    store,
                    llm_client=_NoopLLMClient(),
                    mcp_invoker=invoker,
                )

            self.assertEqual(invoker.calls, [])
            self.assertEqual(state.company_matches, [])
            self.assertEqual(state.company_coverage[0]["coverage_status"], "discovery_error")
            self.assertTrue(state.fallback_used)
            self.assertIn("未调用旧 CNFinancial-first", " ".join(state.uncertainties))
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
