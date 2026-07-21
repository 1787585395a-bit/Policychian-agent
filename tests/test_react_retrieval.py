from __future__ import annotations

import json
import unittest

from policychain.mcp import FakeMCPInvoker
from policychain.tools.mcp_tools import CNFINANCIAL_SERVER
from policychain.tools.react_retrieval import (
    ReActTool,
    build_langchain_tools,
    run_impact_react_search,
    run_react_retrieval,
)


class SequencePlanner:
    def __init__(self, decisions: list[dict[str, object]]) -> None:
        self.decisions = list(decisions)
        self.prompts: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append((system_prompt, user_prompt))
        if not self.decisions:
            raise AssertionError("No ReAct decision left")
        return json.dumps(self.decisions.pop(0), ensure_ascii=False)


class ReActRetrievalTests(unittest.TestCase):
    def test_react_runner_observes_then_refines_query(self) -> None:
        calls: list[dict[str, object]] = []

        def search(arguments: dict[str, object]) -> list[dict[str, object]]:
            calls.append(arguments)
            index = len(calls)
            return [
                {
                    "title": f"结果：{arguments.get('query')}",
                    "summary": "官方解读摘要",
                    "source_url": f"https://example.test/policy-{index}",
                    "server_name": "web-search",
                    "tool_name": "search",
                }
            ]

        planner = SequencePlanner(
            [
                {"thought": "先搜政策名称", "action": "web.search", "arguments": {"query": "人工智能 政策"}},
                {"thought": "结果太宽，补充官方解读", "action": "web.search", "arguments": {"query": "人工智能 政策 官方解读"}},
                {"thought": "证据足够", "action": "finish", "arguments": {}},
            ]
        )

        result = run_react_retrieval(
            "比较人工智能政策",
            [ReActTool("web.search", "search web evidence", search)],
            planner,
            max_steps=4,
        )

        self.assertEqual([call["query"] for call in calls], ["人工智能 政策", "人工智能 政策 官方解读"])
        self.assertEqual(len(result.evidence), 2)
        self.assertEqual(result.traces[1]["thought"], "结果太宽，补充官方解读")
        self.assertIn("Previous observations", planner.prompts[1][1])

    def test_build_langchain_tools_is_optional(self) -> None:
        tools = build_langchain_tools([ReActTool("noop", "no-op", lambda args: [])])

        self.assertIsInstance(tools, list)

    def test_impact_react_search_keeps_full_sector_catalogs_out_of_react_context(self) -> None:
        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [{"name": "软件开发"}],
                (CNFINANCIAL_SERVER, "get_concept_list"): [{"name": "人工智能"}],
                (CNFINANCIAL_SERVER, "search_news"): [{"title": "行业新闻", "summary": "AI 产业新闻"}],
            }
        )
        planner = SequencePlanner([{"thought": "已有 CNFinancial 结果", "action": "finish", "arguments": {}}])

        result = run_impact_react_search("人工智能安全评估", invoker=invoker, llm_client=planner, max_steps=1)

        self.assertEqual([call["tool_name"] for call in invoker.calls], ["search_news"])
        self.assertEqual(result.traces[0]["action"], "cnfinancial.search_news")
        self.assertNotIn("get_industry_list", planner.prompts[0][1])
        self.assertNotIn("get_concept_list", planner.prompts[0][1])
        self.assertTrue(result.evidence)


if __name__ == "__main__":
    unittest.main()
