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

    def test_run_llm_policy_analyst_rejects_empty_policy_identity_id(self) -> None:
        payload = _policy_payload()
        payload["policy_identity"]["policy_id"] = ""
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(payload, ensure_ascii=False))
        try:
            state = PolicyResearchState(user_query="生成式人工智能")

            with self.assertRaisesRegex(LLMPolicyAnalysisError, "got empty"):
                run_llm_policy_analyst(state, store, llm_client=client)
        finally:
            store.close()

    def test_run_llm_policy_analyst_normalizes_evidence_id_proven_by_main_chunk(self) -> None:
        source_policy = _main_source_policy()
        payload = _policy_payload_for_source(source_policy)
        payload["evidence"][0].update(
            {
                "policy_id": "POL-2099-NAT-9999",
                "chunk_id": source_policy["chunks"][0]["chunk_id"],
                "source_url": None,
                "text": "仅为不匹配正文的概括",
            }
        )
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(payload, ensure_ascii=False))
        try:
            state = PolicyResearchState(user_query="主政策分析", source_policy=source_policy)

            output = run_llm_policy_analyst(state, store, llm_client=client)

            self.assertEqual(output.evidence[0].policy_id, source_policy["policy_id"])
            self.assertEqual(state.evidence[0]["policy_id"], source_policy["policy_id"])
            self.assertEqual(state.policy_analysis["evidence"][0]["policy_id"], source_policy["policy_id"])
        finally:
            store.close()

    def test_run_llm_policy_analyst_normalizes_evidence_id_proven_by_main_url(self) -> None:
        source_policy = _main_source_policy()
        payload = _policy_payload_for_source(source_policy)
        payload["evidence"][0].update(
            {
                "policy_id": "POL-2099-NAT-9999",
                "chunk_id": None,
                "source_url": source_policy["source_url"],
                "text": "仅为不匹配正文的概括",
            }
        )
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(payload, ensure_ascii=False))
        try:
            state = PolicyResearchState(user_query="主政策分析", source_policy=source_policy)

            output = run_llm_policy_analyst(state, store, llm_client=client)

            self.assertEqual(output.evidence[0].policy_id, source_policy["policy_id"])
            self.assertEqual(state.evidence[0]["policy_id"], source_policy["policy_id"])
        finally:
            store.close()

    def test_run_llm_policy_analyst_normalizes_evidence_id_proven_by_main_text(self) -> None:
        source_policy = _main_source_policy()
        payload = _policy_payload_for_source(source_policy)
        payload["evidence"][0].update(
            {
                "policy_id": "POL-2099-NAT-9999",
                "chunk_id": None,
                "source_url": None,
                "text": "服务提供者应当依法履行安全义务",
            }
        )
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(payload, ensure_ascii=False))
        try:
            state = PolicyResearchState(user_query="主政策分析", source_policy=source_policy)

            output = run_llm_policy_analyst(state, store, llm_client=client)

            self.assertEqual(output.evidence[0].policy_id, source_policy["policy_id"])
            self.assertEqual(state.evidence[0]["policy_id"], source_policy["policy_id"])
        finally:
            store.close()

    def test_run_llm_policy_analyst_rejects_evidence_policy_id_mismatch(self) -> None:
        source_policy = _main_source_policy()
        payload = _policy_payload_for_source(source_policy)
        payload["evidence"][0].update(
            {
                "policy_id": "POL-2099-NAT-9999",
                "chunk_id": None,
                "source_url": None,
                "text": "无法在主政策正文中核验的摘要",
            }
        )
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(payload, ensure_ascii=False))
        try:
            state = PolicyResearchState(user_query="主政策分析", source_policy=source_policy)

            with self.assertRaisesRegex(LLMPolicyAnalysisError, "evidence policy_id mismatch"):
                run_llm_policy_analyst(state, store, llm_client=client)
        finally:
            store.close()

    def test_run_llm_policy_analyst_rejects_similar_policy_evidence_with_overlapping_text(self) -> None:
        source_policy = _main_source_policy()
        payload = _policy_payload_for_source(source_policy)
        payload["evidence"][0].update(
            {
                "policy_id": "POL-2023-NAT-0048",
                "chunk_id": "POL-2023-NAT-0048-S02-C001",
                "source_url": "https://www.gov.cn/zhengce/zhengceku/202307/content_6891752.htm",
                "text": "服务提供者应当依法履行安全义务",
            }
        )
        store = build_sample_store()
        client = RecordingLLMClient(json.dumps(payload, ensure_ascii=False))
        try:
            state = PolicyResearchState(user_query="主政策分析", source_policy=source_policy)

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


def _main_source_policy() -> dict[str, object]:
    policy_id = "INPUT-ABCDEF12"
    source_url = "https://policy.example.test/main-policy"
    text = (
        "主政策管理办法\n"
        "第一条 为了规范主政策实施，服务提供者应当依法履行安全义务。\n"
        "第二条 主管部门应当建立监督检查和风险处置机制。"
    )
    return {
        "input_type": "text",
        "raw_input": text,
        "content_hash": "abcdef12",
        "policy_id": policy_id,
        "title": "主政策管理办法",
        "source_url": source_url,
        "text": text,
        "metadata": {
            "policy_id": policy_id,
            "title": "主政策管理办法",
            "source_url": source_url,
        },
        "chunks": [
            {
                "policy_id": policy_id,
                "chunk_id": f"{policy_id}-S001-C001",
                "content": text,
            }
        ],
    }


def _policy_payload_for_source(source_policy: dict[str, object]) -> dict[str, object]:
    payload = _policy_payload()
    payload["policy_identity"].update(
        {
            "policy_id": source_policy["policy_id"],
            "title": source_policy["title"],
            "source_url": source_policy["source_url"],
        }
    )
    payload["evidence"][0]["policy_id"] = source_policy["policy_id"]
    return payload


if __name__ == "__main__":
    unittest.main()
