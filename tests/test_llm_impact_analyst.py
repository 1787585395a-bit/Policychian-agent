from __future__ import annotations

import json
import unittest

from policychain.agents import (
    LLMImpactAnalysisError,
    run_llm_impact_analyst,
    run_policy_analyst,
)
from policychain.schemas.agent_outputs import ImpactAnalysisOutput
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


class LLMImpactAnalystTests(unittest.TestCase):
    def test_run_llm_impact_analyst_writes_state_from_valid_json(self) -> None:
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(_impact_payload(), ensure_ascii=False))
        try:
            state = _state_with_policy_analysis(store)
            output = run_llm_impact_analyst(state, llm_client=client)

            self.assertIsInstance(output, ImpactAnalysisOutput)
            self.assertEqual(output.implementation_chain[0].step_index, 1)
            self.assertEqual(state.implementation_path[0]["step_index"], 1)
            self.assertEqual(state.industry_impacts[0]["industry"], "生成式人工智能服务")
            self.assertTrue(state.evidence)
            self.assertEqual(len(client.calls), 1)
        finally:
            store.close()

    def test_run_llm_impact_analyst_prompt_contains_policy_analysis_and_chunks(self) -> None:
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(_impact_payload(), ensure_ascii=False))
        try:
            state = _state_with_policy_analysis(store)
            run_llm_impact_analyst(state, llm_client=client)

            system_prompt, user_prompt = client.calls[0]
            self.assertIn("Impact Analyst", system_prompt)
            self.assertIn("工具层负责取数", system_prompt)
            self.assertIn("POL-2023-NAT-0048", user_prompt)
            self.assertIn("实施链条", user_prompt)
            self.assertIn("只输出一个合法 JSON 对象", user_prompt)
        finally:
            store.close()

    def test_run_llm_impact_analyst_missing_policy_analysis_does_not_call_llm(self) -> None:
        client = RecordingLLMClient(json.dumps(_impact_payload(), ensure_ascii=False))
        state = PolicyResearchState(user_query="生成式人工智能")

        output = run_llm_impact_analyst(state, llm_client=client)

        self.assertEqual(output.implementation_chain, [])
        self.assertEqual(output.industry_impacts, [])
        self.assertEqual(client.calls, [])
        self.assertTrue(state.uncertainties)

    def test_run_llm_impact_analyst_rejects_malformed_json(self) -> None:
        store = build_sample_store()
        client = RecordingLLMClient("not json")
        try:
            state = _state_with_policy_analysis(store)

            with self.assertRaises(StructuredOutputError):
                run_llm_impact_analyst(state, llm_client=client)
        finally:
            store.close()

    def test_run_llm_impact_analyst_rejects_top_level_evidence_policy_id_mismatch(self) -> None:
        payload = _impact_payload()
        payload["evidence"][0]["policy_id"] = "POL-2099-NAT-9999"
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(payload, ensure_ascii=False))
        try:
            state = _state_with_policy_analysis(store)

            with self.assertRaisesRegex(LLMImpactAnalysisError, "evidence policy_id mismatch"):
                run_llm_impact_analyst(state, llm_client=client)
        finally:
            store.close()

    def test_run_llm_impact_analyst_rejects_nested_evidence_policy_id_mismatch(self) -> None:
        payload = _impact_payload()
        payload["industry_impacts"][0]["evidence"][0]["policy_id"] = "POL-2099-NAT-9999"
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(payload, ensure_ascii=False))
        try:
            state = _state_with_policy_analysis(store)

            with self.assertRaisesRegex(LLMImpactAnalysisError, "evidence policy_id mismatch"):
                run_llm_impact_analyst(state, llm_client=client)
        finally:
            store.close()


def _state_with_policy_analysis(store) -> PolicyResearchState:
    state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
    run_policy_analyst(state, store)
    return state


def _impact_payload() -> dict[str, object]:
    return {
        "implementation_actors": ["生成式人工智能服务提供者"],
        "implementation_mechanisms": ["备案与合规审查"],
        "implementation_chain": [
            {
                "step_index": 1,
                "actor": "生成式人工智能服务提供者",
                "action": "落实内容安全要求",
                "mechanism": "备案与合规审查",
                "evidence": [_evidence()],
            }
        ],
        "industry_impacts": [
            {
                "industry": "生成式人工智能服务",
                "impact_type": "direct",
                "direction": "mixed",
                "transmission_logic": "政策直接影响服务提供和安全治理能力建设",
                "conditions": ["需结合监管执行口径"],
                "risks": ["合规能力不足会提高整改压力"],
                "evidence": [_evidence()],
            }
        ],
        "uncertainties": ["尚未接入产业数据"],
        "evidence": [_evidence()],
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
