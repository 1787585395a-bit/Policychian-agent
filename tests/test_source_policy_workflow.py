from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from policychain.agents.policy_analyst import PolicyAnalysisError
from policychain.agents import run_llm_policy_analyst, run_policy_analyst
from policychain.graph import run_policy_research_workflow
from policychain.source_policy import build_source_policy_from_text
from policychain.state import PolicyResearchState
from tests.helpers import build_sample_store


UNRELATED_POLICY_TEXT = """量子算力基础设施管理办法

第一条 为了规范量子算力基础设施建设和运营，维护公共数据安全，制定本办法。
第二条 量子算力服务提供者应当建立安全管理制度，依法履行数据安全保护义务。
第三条 主管部门应当加强监督管理，推动行业组织建立服务能力评估机制。
"""

AI_POLICY_TEXT = """生成式人工智能服务管理办法

第一条 为了促进生成式人工智能健康发展和规范应用，维护国家安全和社会公共利益，制定本办法。
第二条 生成式人工智能服务提供者应当依法承担网络信息内容生产者责任。
第三条 服务提供者应当开展算法模型安全评估，提升训练数据质量。
"""


class RecordingLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


class SourcePolicyWorkflowTests(unittest.TestCase):
    def test_user_policy_text_is_primary_even_without_similar_policy(self) -> None:
        store = build_sample_store()
        try:
            state = run_policy_research_workflow(UNRELATED_POLICY_TEXT, store)

            self.assertTrue(state.policy_analysis["policy_identity"]["policy_id"].startswith("INPUT-"))
            self.assertEqual(state.similar_policy_matches, [])
            self.assertIn("未在本地知识库中找到相似政策", state.final_report)
            self.assertTrue(state.policy_analysis["policy_identity"]["title"])
            self.assertGreaterEqual(len(state.implementation_path), 1)
            self.assertGreaterEqual(len(state.industry_impacts), 1)
            self.assertTrue(state.progress_events)
            self.assertTrue(any(event["progress"] == 15 for event in state.progress_events))
            stages = {event["stage"] for event in state.progress_events}
            self.assertIn("URL 抓取", stages)
            self.assertIn("正文质量校验", stages)
        finally:
            store.close()

    def test_similar_policy_matches_do_not_replace_source_policy_identity(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query=AI_POLICY_TEXT)
            output = run_policy_analyst(state, store)

            self.assertTrue(output.policy_identity["policy_id"].startswith("INPUT-"))
            self.assertTrue(state.similar_policy_matches)
            self.assertTrue(all(match["policy_id"] != output.policy_identity["policy_id"] for match in state.similar_policy_matches))
        finally:
            store.close()

    def test_llm_policy_prompt_contains_source_policy_and_similar_matches(self) -> None:
        store = build_sample_store()
        source_policy = build_source_policy_from_text(AI_POLICY_TEXT, raw_input=AI_POLICY_TEXT, input_type="text")
        policy_id = source_policy["policy_id"]
        client = RecordingLLMClient(json.dumps(_policy_payload(policy_id), ensure_ascii=False))
        try:
            state = PolicyResearchState(user_query=AI_POLICY_TEXT, source_policy=source_policy)
            run_llm_policy_analyst(state, store, llm_client=client)

            user_prompt = client.calls[0][1]
            self.assertIn("用户输入的主政策", user_prompt)
            self.assertIn("本地知识库相似政策", user_prompt)
            self.assertEqual(state.policy_analysis["policy_identity"]["policy_id"], policy_id)
            self.assertTrue(state.similar_policy_matches)
        finally:
            store.close()

    def test_invalid_url_does_not_fallback_to_unknown_report(self) -> None:
        store = build_sample_store()
        events: list[tuple[int, str, str]] = []
        try:
            with patch("policychain.source_policy.urlopen", side_effect=OSError("network refused")):
                with self.assertRaisesRegex(PolicyAnalysisError, "Failed to fetch policy URL"):
                    run_policy_analyst(
                        PolicyResearchState(user_query="https://example.test/not-policy"),
                        store,
                        progress_callback=lambda progress, stage, message: events.append((progress, stage, message)),
                    )

            stages = {stage for _, stage, _ in events}
            self.assertIn("URL 抓取", stages)
            self.assertIn("正文质量校验", stages)
        finally:
            store.close()


def _policy_payload(policy_id: str) -> dict[str, object]:
    return {
        "policy_identity": {
            "policy_id": policy_id,
            "title": "生成式人工智能服务管理办法",
            "source_url": None,
        },
        "policy_goals": ["规范生成式人工智能服务"],
        "target_entities": ["生成式人工智能服务提供者"],
        "policy_measures": ["服务提供者应当开展算法模型安全评估"],
        "historical_changes": [],
        "strength_assessment": {
            "level": "medium",
            "reasons": ["文本包含明确义务要求"],
            "uncertainties": [],
        },
        "evidence": [
            {
                "policy_id": policy_id,
                "chunk_id": f"{policy_id}-S001-C001",
                "source_url": None,
                "text": "服务提供者应当开展算法模型安全评估",
                "note": "正文",
            }
        ],
        "uncertainties": [],
    }


if __name__ == "__main__":
    unittest.main()
