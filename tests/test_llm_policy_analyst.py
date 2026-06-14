from __future__ import annotations

import json
import unittest

from policychain.agents import LLMPolicyAnalysisError, run_llm_policy_analyst
from policychain.schemas.agent_outputs import PolicyAnalysisOutput
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


class LLMPolicyAnalystTests(unittest.TestCase):
    def test_run_llm_policy_analyst_writes_state_from_valid_json(self) -> None:
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(_policy_payload(), ensure_ascii=False))
        try:
            state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
            output = run_llm_policy_analyst(state, store, llm_client=client)

            self.assertIsInstance(output, PolicyAnalysisOutput)
            self.assertEqual(output.policy_identity["policy_id"], "POL-2023-NAT-0048")
            self.assertEqual(state.policy_ids, ["POL-2023-NAT-0048"])
            self.assertEqual(state.policy_analysis["policy_identity"]["policy_id"], "POL-2023-NAT-0048")
            self.assertGreaterEqual(len(state.policy_chunks), 1)
            self.assertEqual(state.evidence[0]["policy_id"], "POL-2023-NAT-0048")
            self.assertEqual(len(client.calls), 1)
        finally:
            store.close()

    def test_run_llm_policy_analyst_prompt_contains_retrieved_evidence(self) -> None:
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(_policy_payload(), ensure_ascii=False))
        try:
            state = PolicyResearchState(user_query="生成式人工智能")
            run_llm_policy_analyst(state, store, llm_client=client)

            system_prompt, user_prompt = client.calls[0]
            self.assertIn("Policy Analyst", system_prompt)
            self.assertIn("用户输入的主政策", system_prompt)
            self.assertIn("不得用相似政策替代主政策", system_prompt)
            self.assertIn("生成式人工智能", user_prompt)
            self.assertIn("POL-2023-NAT-0048", user_prompt)
            self.assertIn("本地知识库相似政策", user_prompt)
            self.assertIn("只输出一个合法 JSON 对象", user_prompt)
        finally:
            store.close()

    def test_run_llm_policy_analyst_no_search_result_does_not_call_llm(self) -> None:
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(_policy_payload(), ensure_ascii=False))
        try:
            state = PolicyResearchState(user_query="完全不存在的查询词")
            output = run_llm_policy_analyst(state, store, llm_client=client)

            self.assertEqual(output.policy_identity["status"], "no_policy_found")
            self.assertEqual(client.calls, [])
            self.assertTrue(state.uncertainties)
        finally:
            store.close()

    def test_run_llm_policy_analyst_rejects_malformed_json(self) -> None:
        store = build_sample_store()
        client = RecordingLLMClient("not json")
        try:
            state = PolicyResearchState(user_query="生成式人工智能")

            with self.assertRaises(StructuredOutputError):
                run_llm_policy_analyst(state, store, llm_client=client)
        finally:
            store.close()

    def test_run_llm_policy_analyst_rejects_policy_id_mismatch(self) -> None:
        payload = _policy_payload()
        payload["policy_identity"]["policy_id"] = "POL-2099-NAT-9999"
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(payload, ensure_ascii=False))
        try:
            state = PolicyResearchState(user_query="生成式人工智能")

            with self.assertRaisesRegex(LLMPolicyAnalysisError, "policy_id mismatch"):
                run_llm_policy_analyst(state, store, llm_client=client)
        finally:
            store.close()

    def test_run_llm_policy_analyst_rejects_evidence_policy_id_mismatch(self) -> None:
        payload = _policy_payload()
        payload["evidence"][0]["policy_id"] = "POL-2099-NAT-9999"
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(payload, ensure_ascii=False))
        try:
            state = PolicyResearchState(user_query="生成式人工智能")

            with self.assertRaisesRegex(LLMPolicyAnalysisError, "evidence policy_id mismatch"):
                run_llm_policy_analyst(state, store, llm_client=client)
        finally:
            store.close()


def _policy_payload() -> dict[str, object]:
    return {
        "policy_identity": {
            "policy_id": "POL-2023-NAT-0048",
            "title": "生成式人工智能服务管理暂行办法",
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
        "evidence": [
            {
                "policy_id": "POL-2023-NAT-0048",
                "chunk_id": "POL-2023-NAT-0048-S001-C001",
                "source_url": "https://example.test/policy",
                "text": "服务提供者应当依法履行安全义务",
                "note": "第一条",
            }
        ],
        "uncertainties": ["仅基于样例政策文本"],
    }


if __name__ == "__main__":
    unittest.main()
