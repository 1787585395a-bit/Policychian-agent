from __future__ import annotations

import json
import unittest

from policychain.graph import run_llm_policy_research_workflow, run_policy_research_workflow
from tests.helpers import build_sample_store


PROHIBITED_TERMS = ("买入", "卖出", "目标价", "推荐股票")


class SequenceLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self.responses:
            raise AssertionError("No LLM response left")
        return self.responses.pop(0)


class LLMWorkflowTests(unittest.TestCase):
    def test_run_llm_policy_research_workflow_runs_three_llm_agents_and_report(self) -> None:
        store = build_sample_store()
        client = SequenceLLMClient(
            [
                json.dumps(_policy_payload(), ensure_ascii=False),
                json.dumps(_impact_payload(), ensure_ascii=False),
                json.dumps(_company_payload(), ensure_ascii=False),
            ]
        )
        try:
            state = run_llm_policy_research_workflow(
                "生成式人工智能服务提供者有哪些管理要求",
                store,
                llm_client=client,
            )

            self.assertEqual(len(client.calls), 3)
            self.assertEqual(state.policy_analysis["policy_identity"]["policy_id"], "POL-2023-NAT-0048")
            self.assertEqual(state.industry_impacts[0]["industry"], "算法模型研发与评估")
            self.assertEqual(state.company_matches[0]["company_name"], "清源模型安全科技")
            self.assertIn("PolicyChain 政策研究报告", state.final_report)
            self.assertIn("公司业务匹配清单", state.final_report)
            for term in PROHIBITED_TERMS:
                self.assertNotIn(term, state.final_report)
        finally:
            store.close()

    def test_run_llm_policy_research_workflow_no_policy_does_not_call_llm(self) -> None:
        store = build_sample_store()
        client = SequenceLLMClient([])
        try:
            state = run_llm_policy_research_workflow("完全不存在的查询词", store, llm_client=client)

            self.assertEqual(client.calls, [])
            self.assertEqual(state.policy_analysis["policy_identity"]["status"], "no_policy_found")
            self.assertEqual(state.company_matches, [])
            self.assertIn("未检索到可用于政策分析", state.final_report)
        finally:
            store.close()

    def test_default_workflow_remains_deterministic(self) -> None:
        store = build_sample_store()
        try:
            state = run_policy_research_workflow("生成式人工智能服务提供者有哪些管理要求", store)

            self.assertTrue(state.policy_analysis)
            self.assertTrue(state.industry_impacts)
            self.assertTrue(state.company_matches)
            self.assertIn("PolicyChain 政策研究报告", state.final_report)
        finally:
            store.close()


def _policy_payload() -> dict[str, object]:
    return {
        "policy_identity": {
            "policy_id": "POL-2023-NAT-0048",
            "title": "生成式人工智能服务管理暂行办法",
            "document_number": "第15号",
            "publish_date": "2023-05-23",
            "issuing_agencies": ["国家级政府或主管部门（按官方来源）"],
            "source_url": "https://example.test/policy",
        },
        "policy_goals": ["规范生成式人工智能服务"],
        "target_entities": ["生成式人工智能服务提供者"],
        "policy_measures": ["服务提供者应当依法履行安全义务"],
        "historical_changes": [],
        "strength_assessment": {
            "level": "medium",
            "reasons": ["文本包含明确义务要求"],
            "uncertainties": ["尚未纳入配套细则"],
        },
        "evidence": [_evidence()],
        "uncertainties": ["仅基于样例政策文本"],
    }


def _impact_payload() -> dict[str, object]:
    return {
        "implementation_actors": ["生成式人工智能服务提供者"],
        "implementation_mechanisms": ["算法模型治理"],
        "implementation_chain": [
            {
                "step_index": 1,
                "actor": "生成式人工智能服务提供者",
                "action": "落实模型、算法和训练数据治理",
                "mechanism": "算法模型治理",
                "evidence": [_evidence()],
            }
        ],
        "industry_impacts": [
            {
                "industry": "算法模型研发与评估",
                "impact_type": "direct",
                "direction": "mixed",
                "transmission_logic": "政策要求模型、算法和训练数据环节承担安全治理责任。",
                "conditions": ["需结合监管执行口径"],
                "risks": ["合规能力不足会提高整改压力"],
                "evidence": [_evidence()],
            }
        ],
        "uncertainties": ["尚未接入产业数据"],
        "evidence": [_evidence()],
    }


def _company_payload() -> dict[str, object]:
    return {
        "companies": [
            {
                "company_name": "清源模型安全科技",
                "industry_segment": "算法模型研发与评估",
                "matched_business": "提供大模型安全评估、训练数据质量检测和模型风险测试服务。",
                "match_level": "high",
                "business_evidence": [
                    {
                        "source_name": "Mock Company Profile",
                        "source_url": "mock://company/qingyuan-model-safety",
                        "text": "公司资料显示其核心服务包括模型安全评估、训练数据质量检测和生成式人工智能风险测试。",
                        "data_date": "2026-01-15",
                    }
                ],
                "policy_link": "政策要求模型、算法和训练数据环节承担安全治理责任。",
                "revenue_relevance": "medium",
                "conditions": ["需核验真实官网、年报和公告资料。"],
                "risks": ["本地 mock 数据不可代表真实公司资料。"],
                "data_date": "2026-01-15",
                "confidence": 0.86,
            }
        ],
        "uncertainties": ["公司资料来自本地 mock 数据，仅用于验证流程。"],
    }


def _evidence() -> dict[str, object]:
    return {
        "policy_id": "POL-2023-NAT-0048",
        "chunk_id": "POL-2023-NAT-0048-S001-C001",
        "source_url": "https://example.test/policy",
        "text": "服务提供者应当依法履行安全义务",
        "note": "第一条",
    }


if __name__ == "__main__":
    unittest.main()
