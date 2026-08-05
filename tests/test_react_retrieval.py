from __future__ import annotations

import json
import unittest

from policychain.mcp import FakeMCPInvoker
from policychain.tools.mcp_tools import (
    CNFINANCIAL_SERVER,
    _invoke_with_log,
    candidate_retrieval_statuses,
    collect_company_candidates,
)
from policychain.tools.react_retrieval import (
    ReActTool,
    build_langchain_tools,
    run_company_react_search,
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

    def test_company_stock_search_budget_is_shared_across_all_retrieval_routes(self) -> None:
        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [],
                (CNFINANCIAL_SERVER, "get_concept_list"): [],
                (CNFINANCIAL_SERVER, "search_stock"): [],
            }
        )
        impact = {
            "impact_id": "IMP-001",
            "industry": "海水淡化设备",
            "chain_segment": "反渗透膜组件",
            "business_variables": ["高压泵"],
            "affected_company_types": ["海水淡化设备供应商"],
        }
        logs: list[dict[str, object]] = []

        collect_company_candidates([impact], invoker=invoker, tool_logs=logs)
        planner = SequencePlanner(
            [
                {
                    "thought": "尝试模型生成的长路径查询",
                    "action": "cnfinancial.search_stock",
                    "arguments": {"query": "政策措施→项目建设→海水淡化设备订单与膜组件需求"},
                }
            ]
        )
        run_company_react_search(
            "政策措施→项目建设→海水淡化设备订单与膜组件需求",
            invoker=invoker,
            llm_client=planner,
            max_steps=1,
            impact=impact,
            impact_id="IMP-001",
            tool_logs=logs,
        )
        _invoke_with_log(
            invoker,
            CNFINANCIAL_SERVER,
            "search_stock",
            {"keyword": "海水淡化膜"},
            tool_logs=logs,
            log_context={"impact_id": "IMP-001", "source_type": "fallback"},
        )
        _invoke_with_log(
            invoker,
            CNFINANCIAL_SERVER,
            "search_stock",
            {"keyword": "液冷服务器"},
            tool_logs=logs,
            log_context={"impact_id": "IMP-002", "source_type": "fallback"},
        )

        calls_by_query = [
            str(call["arguments"]["keyword"])
            for call in invoker.calls
            if call["tool_name"] == "search_stock"
        ]
        self.assertEqual(len(calls_by_query), 3)
        self.assertEqual(calls_by_query[-1], "液冷服务器")
        skipped = [log for log in logs if log.get("tool_name") == "search_stock" and log.get("status") == "skipped"]
        self.assertTrue(any(log.get("skip_reason") == "query_budget" for log in skipped))
        self.assertTrue(any(log.get("skip_reason") == "invalid_query" for log in skipped))
        self.assertTrue(all(log.get("impact_id") and log.get("query") for log in skipped))
        self.assertNotIn("政策措施→项目建设", planner.prompts[0][1])
        self.assertIn("反渗透膜组件", planner.prompts[0][1])
        status = candidate_retrieval_statuses(logs)["IMP-001"]
        self.assertEqual(status["query_count"], 2)
        self.assertGreaterEqual(status["skipped_query_count"], 3)

    def test_company_stock_search_final_boundary_rejects_long_or_descriptive_queries_without_spending_budget(self) -> None:
        invoker = FakeMCPInvoker({(CNFINANCIAL_SERVER, "search_stock"): []})
        logs: list[dict[str, object]] = []
        queries = [
            "政策支持数据中心建设并推动企业降低单位算力能耗",
            "算力基础设施→服务器采购→能源需求",
            "完整政策描述" * 50,
            "液冷服务器",
        ]

        for query in queries:
            _invoke_with_log(
                invoker,
                CNFINANCIAL_SERVER,
                "search_stock",
                {"keyword": query},
                tool_logs=logs,
                log_context={"impact_id": "IMP-001", "source_type": "test"},
            )

        calls = [call for call in invoker.calls if call["tool_name"] == "search_stock"]
        self.assertEqual([call["arguments"]["keyword"] for call in calls], ["液冷服务器"])
        self.assertEqual(
            [log.get("skip_reason") for log in logs if log.get("status") == "skipped"],
            ["invalid_query", "invalid_query", "invalid_query"],
        )
        self.assertEqual(logs[-1].get("query_budget_used"), 1)

    def test_company_stock_search_rejects_metrics_operating_variables_and_generic_chain_terms(self) -> None:
        invoker = FakeMCPInvoker({(CNFINANCIAL_SERVER, "search_stock"): []})
        logs: list[dict[str, object]] = []
        invalid_queries = [
            "工程收入确认节奏",
            "关键装备销量和单价",
            "新增海水淡化设施投资额",
            "能效与碳效指标",
            "绿电交易量",
            "运营效率",
            "配套率",
            "应用场景",
            "关键装备制造环节",
        ]

        for query in [*invalid_queries, "反渗透膜", "高压泵"]:
            _invoke_with_log(
                invoker,
                CNFINANCIAL_SERVER,
                "search_stock",
                {"keyword": query},
                tool_logs=logs,
                log_context={"impact_id": "IMP-001", "source_type": "test"},
            )
        for query in ("能量回收装置", "液冷服务器"):
            _invoke_with_log(
                invoker,
                CNFINANCIAL_SERVER,
                "search_stock",
                {"keyword": query},
                tool_logs=logs,
                log_context={"impact_id": "IMP-002", "source_type": "test"},
            )
        _invoke_with_log(
            invoker,
            CNFINANCIAL_SERVER,
            "search_stock",
            {"keyword": "数据中心"},
            tool_logs=logs,
            log_context={"impact_id": "IMP-003", "source_type": "test"},
        )

        calls = [call for call in invoker.calls if call["tool_name"] == "search_stock"]
        self.assertEqual(
            [call["arguments"]["keyword"] for call in calls],
            ["反渗透膜", "高压泵", "能量回收装置", "液冷服务器", "数据中心"],
        )
        invalid_logs = [log for log in logs if log.get("skip_reason") == "invalid_query"]
        self.assertEqual([log.get("query") for log in invalid_logs], invalid_queries)
        self.assertTrue(all(log.get("status") == "skipped" for log in invalid_logs))
        self.assertEqual(logs[-1].get("query_budget_used"), 1)


if __name__ == "__main__":
    unittest.main()
