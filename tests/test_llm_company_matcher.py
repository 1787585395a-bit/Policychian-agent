from __future__ import annotations

import json
import unittest

from policychain.agents import (
    LLMCompanyMatchError,
    run_company_matcher,
    run_impact_analyst,
    run_llm_company_matcher,
    run_policy_analyst,
)
from policychain.schemas.agent_outputs import CompanyMatchOutput
from policychain.state import PolicyResearchState
from policychain.structured_output import StructuredOutputError
from tests.helpers import build_sample_store


class RecordingLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


class LLMCompanyMatcherTests(unittest.TestCase):
    def test_run_llm_company_matcher_writes_state_from_valid_json(self) -> None:
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(_company_payload(), ensure_ascii=False))
        try:
            state = _state_with_industry_impacts(store)
            output = run_llm_company_matcher(state, llm_client=client)

            self.assertIsInstance(output, CompanyMatchOutput)
            self.assertEqual(output.companies[0].company_name, "清源模型安全科技")
            self.assertGreaterEqual(len(state.company_candidates), 1)
            self.assertEqual(state.company_matches[0]["company_name"], "清源模型安全科技")
            self.assertTrue(state.uncertainties)
            self.assertEqual(len(client.calls), 1)
        finally:
            store.close()

    def test_run_llm_company_matcher_prompt_contains_impacts_and_company_candidates(self) -> None:
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(_company_payload(), ensure_ascii=False))
        try:
            state = _state_with_industry_impacts(store)
            run_llm_company_matcher(state, llm_client=client)

            system_prompt, user_prompt = client.calls[0]
            self.assertIn("Company Matcher", system_prompt)
            self.assertIn("业务相关性匹配", system_prompt)
            self.assertIn("生成式人工智能服务", user_prompt)
            self.assertIn("清源模型安全科技", user_prompt)
            self.assertIn("只输出一个合法 JSON 对象", user_prompt)
        finally:
            store.close()

    def test_run_llm_company_matcher_missing_impacts_does_not_call_llm(self) -> None:
        client = RecordingLLMClient(json.dumps(_company_payload(), ensure_ascii=False))
        state = PolicyResearchState(user_query="生成式人工智能")

        output = run_llm_company_matcher(state, llm_client=client)

        self.assertEqual(output.companies, [])
        self.assertEqual(state.company_candidates, [])
        self.assertEqual(state.company_matches, [])
        self.assertEqual(client.calls, [])
        self.assertTrue(state.uncertainties)

    def test_run_llm_company_matcher_no_company_records_does_not_call_llm(self) -> None:
        client = RecordingLLMClient(json.dumps(_company_payload(), ensure_ascii=False))
        state = PolicyResearchState(
            user_query="无匹配行业",
            industry_impacts=[
                {
                    "industry": "无匹配行业",
                    "transmission_logic": "没有本地公司资料覆盖",
                    "conditions": [],
                    "risks": [],
                }
            ],
        )

        output = run_llm_company_matcher(state, llm_client=client)

        self.assertEqual(output.companies, [])
        self.assertEqual(client.calls, [])
        self.assertTrue(state.uncertainties)

    def test_run_llm_company_matcher_rejects_malformed_json(self) -> None:
        store = build_sample_store()
        client = RecordingLLMClient("not json")
        try:
            state = _state_with_industry_impacts(store)

            with self.assertRaises(StructuredOutputError):
                run_llm_company_matcher(state, llm_client=client)
        finally:
            store.close()

    def test_run_llm_company_matcher_rejects_company_outside_candidates(self) -> None:
        payload = _company_payload()
        payload["companies"][0]["company_name"] = "不存在的公司"
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(payload, ensure_ascii=False))
        try:
            state = _state_with_industry_impacts(store)

            with self.assertRaisesRegex(LLMCompanyMatchError, "outside candidate records"):
                run_llm_company_matcher(state, llm_client=client)
        finally:
            store.close()

    def test_llm_and_deterministic_company_matcher_keep_default_runner_separate(self) -> None:
        store = build_sample_store()
        try:
            deterministic_state = _state_with_industry_impacts(store)
            llm_state = PolicyResearchState(
                user_query=deterministic_state.user_query,
                policy_ids=list(deterministic_state.policy_ids),
                policy_chunks=list(deterministic_state.policy_chunks),
                policy_analysis=dict(deterministic_state.policy_analysis),
                implementation_path=list(deterministic_state.implementation_path),
                industry_impacts=list(deterministic_state.industry_impacts),
                evidence=list(deterministic_state.evidence),
                uncertainties=list(deterministic_state.uncertainties),
            )
            run_company_matcher(deterministic_state)
            run_llm_company_matcher(
                llm_state,
                llm_client=RecordingLLMClient(json.dumps(_company_payload(), ensure_ascii=False)),
            )

            self.assertTrue(deterministic_state.company_matches)
            self.assertTrue(llm_state.company_matches)
            self.assertEqual(llm_state.company_matches[0]["company_name"], "清源模型安全科技")
        finally:
            store.close()


def _state_with_industry_impacts(store) -> PolicyResearchState:
    state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
    run_policy_analyst(state, store)
    run_impact_analyst(state)
    return state


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


if __name__ == "__main__":
    unittest.main()
