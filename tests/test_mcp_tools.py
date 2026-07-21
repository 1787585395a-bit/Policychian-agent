from __future__ import annotations

import unittest
from unittest.mock import patch

from policychain.mcp import FakeMCPInvoker
from policychain.tools.mcp_tools import (
    CNFINANCIAL_SERVER,
    OPEN_WEBSEARCH_SEARCH_TOOL,
    OPEN_WEBSEARCH_SERVER,
    collect_company_candidates,
    collect_company_web_evidence,
    collect_impact_research,
    fetch_web_content,
    merge_react_company_candidates,
    search_web,
)


class MCPToolsTests(unittest.TestCase):
    def test_search_web_normalizes_fake_invoker_results(self) -> None:
        invoker = FakeMCPInvoker(
            {
                (OPEN_WEBSEARCH_SERVER, OPEN_WEBSEARCH_SEARCH_TOOL): [
                    {
                        "title": "官方解读",
                        "source": "国务院",
                        "date": "2026-01-02",
                        "url": "https://example.test/policy",
                        "description": "政策实施细则摘要",
                    }
                ]
            }
        )

        evidence = search_web("生成式人工智能 官方解读", source_priority=["政府官网"], top_k=1, invoker=invoker)

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["title"], "官方解读")
        self.assertEqual(evidence[0]["source_url"], "https://example.test/policy")
        self.assertEqual(evidence[0]["tool_name"], OPEN_WEBSEARCH_SEARCH_TOOL)
        self.assertTrue(evidence[0]["query_time"])

    def test_unavailable_invoker_returns_empty_results(self) -> None:
        self.assertEqual(search_web("生成式人工智能", top_k=1), [])
        self.assertEqual(fetch_web_content("https://example.test/policy"), [])

    def test_mcp_error_payload_returns_empty_and_records_error(self) -> None:
        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "search_news"): {
                    "result": {
                        "error": True,
                        "message": "CNFinancial proxy failed",
                    }
                }
            }
        )

        research = collect_impact_research({}, [_ai_impact()], invoker=invoker, top_k=1)

        self.assertTrue(any(item["tool_name"] == "select_sectors" for item in research["cnfinancial"]))
        self.assertFalse(any(item["tool_name"] == "search_news" for item in research["cnfinancial"]))
        self.assertTrue(any(log["status"] == "error" for log in research["tool_logs"]))
        self.assertTrue(any("CNFinancial proxy failed" in log["error"] for log in research["tool_logs"]))
        self.assertTrue(any("CNFinancial proxy failed" in error for error in getattr(invoker, "errors", [])))

    def test_collect_impact_research_uses_cnfinancial_official_arguments(self) -> None:
        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [
                    {"\u540d\u79f0": "\u8f6f\u4ef6\u5f00\u53d1"},
                    {"\u540d\u79f0": "\u7164\u70ad\u884c\u4e1a"},
                ],
                (CNFINANCIAL_SERVER, "get_concept_list"): [
                    {"\u540d\u79f0": "\u4eba\u5de5\u667a\u80fd"},
                ],
                (CNFINANCIAL_SERVER, "get_industry_stocks"): [
                    {
                        "\u540d\u79f0": "\u793a\u4f8b\u79d1\u6280",
                        "\u4ee3\u7801": "300001",
                    }
                ],
                (CNFINANCIAL_SERVER, "search_news"): [
                    {
                        "title": "AI compliance news",
                        "description": "industry news",
                    }
                ],
            }
        )

        research = collect_impact_research({}, [_ai_impact()], invoker=invoker, top_k=1)

        industry_call = next(call for call in invoker.calls if call["tool_name"] == "get_industry_stocks")
        news_call = next(call for call in invoker.calls if call["tool_name"] == "search_news")
        fund_flow_call = next(call for call in invoker.calls if call["tool_name"] == "get_sector_fund_flow")

        self.assertEqual(set(industry_call["arguments"]), {"industry"})
        self.assertIn(
            industry_call["arguments"]["industry"],
            {"\u8f6f\u4ef6\u5f00\u53d1", "\u4e92\u8054\u7f51\u670d\u52a1", "\u8ba1\u7b97\u673a\u8bbe\u5907"},
        )
        self.assertEqual(set(news_call["arguments"]), {"keyword", "num_results"})
        self.assertEqual(set(fund_flow_call["arguments"]), {"sector_type", "indicator"})
        self.assertTrue(any(log["server_name"] == CNFINANCIAL_SERVER for log in research["tool_logs"]))
        self.assertTrue(any(log["tool_name"] == "get_industry_stocks" and log["count"] == 1 for log in research["tool_logs"]))
        self.assertTrue(any(log["tool_name"] == "select_sectors" and log["count"] >= 1 for log in research["tool_logs"]))

    def test_collect_company_candidates_prefers_industry_stocks_and_enriches_by_symbol(self) -> None:
        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [
                    {"\u540d\u79f0": "\u8f6f\u4ef6\u5f00\u53d1"},
                    {"\u540d\u79f0": "\u7164\u70ad\u884c\u4e1a"},
                ],
                (CNFINANCIAL_SERVER, "get_concept_list"): [
                    {"\u540d\u79f0": "\u4eba\u5de5\u667a\u80fd"},
                ],
                (CNFINANCIAL_SERVER, "get_industry_stocks"): [
                    {
                        "\u540d\u79f0": "\u793a\u4f8b\u79d1\u6280",
                        "\u4ee3\u7801": "300001",
                        "\u4e3b\u8425\u4e1a\u52a1": "\u4eba\u5de5\u667a\u80fd\u8f6f\u4ef6\u670d\u52a1",
                    }
                ],
                (CNFINANCIAL_SERVER, "get_company_profile"): [
                    {
                        "\u4e3b\u8425\u4e1a\u52a1": "\u6a21\u578b\u5b89\u5168\u8bc4\u4f30\u5e73\u53f0",
                        "\u6536\u5165\u5360\u6bd4": "28%",
                    }
                ],
            }
        )

        tool_logs: list[dict[str, object]] = []
        candidates = collect_company_candidates([_ai_impact()], invoker=invoker, top_k_per_industry=1, tool_logs=tool_logs)

        self.assertEqual(candidates[0]["company_name"], "\u793a\u4f8b\u79d1\u6280")
        self.assertEqual(candidates[0]["stock_code"], "300001")
        self.assertEqual(candidates[0]["revenue_relevance"], "28%")
        self.assertEqual(candidates[0]["candidate_source_tool"], "get_industry_stocks")
        self.assertEqual(candidates[0]["candidate_query"], "\u8f6f\u4ef6\u5f00\u53d1")
        industry_call = next(call for call in invoker.calls if call["tool_name"] == "get_industry_stocks")
        profile_call = next(call for call in invoker.calls if call["tool_name"] == "get_company_profile")
        self.assertEqual(set(industry_call["arguments"]), {"industry"})
        self.assertEqual(profile_call["arguments"], {"symbol": "300001"})
        self.assertTrue(any(log["tool_name"] == "get_industry_stocks" and log["count"] == 1 for log in tool_logs))
        self.assertTrue(any(log["tool_name"] == "get_company_profile" and log["count"] == 1 for log in tool_logs))

    def test_collect_company_candidates_does_not_use_concept_as_industry_board(self) -> None:
        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [
                    {"\u540d\u79f0": "\u8f6f\u4ef6\u5f00\u53d1"},
                ],
                (CNFINANCIAL_SERVER, "get_concept_list"): [
                    {"\u540d\u79f0": "ChatGPT\u6982\u5ff5"},
                ],
                (CNFINANCIAL_SERVER, "search_stock"): [
                    {
                        "\u540d\u79f0": "\u6982\u5ff5\u79d1\u6280",
                        "\u4ee3\u7801": "300002",
                    }
                ],
            }
        )

        tool_logs: list[dict[str, object]] = []
        candidates = collect_company_candidates([_ai_impact()], invoker=invoker, top_k_per_industry=1, tool_logs=tool_logs)

        industry_calls = [call for call in invoker.calls if call["tool_name"] == "get_industry_stocks"]
        self.assertTrue(all(call["arguments"]["industry"] != "ChatGPT\u6982\u5ff5" for call in industry_calls))
        self.assertTrue(any(call["tool_name"] == "search_stock" for call in invoker.calls))
        self.assertTrue(candidates)

    def test_fast_mode_skips_expensive_impact_tools(self) -> None:
        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [
                    {"\u540d\u79f0": "\u8f6f\u4ef6\u5f00\u53d1"},
                ],
                (CNFINANCIAL_SERVER, "get_concept_list"): [
                    {"\u540d\u79f0": "\u4eba\u5de5\u667a\u80fd"},
                ],
                (CNFINANCIAL_SERVER, "get_industry_stocks"): [
                    {"\u540d\u79f0": "\u793a\u4f8b\u79d1\u6280", "\u4ee3\u7801": "300001"},
                ],
            }
        )

        with patch.dict("os.environ", {"POLICYCHAIN_MCP_FAST_MODE": "1"}):
            research = collect_impact_research({}, [_ai_impact()], invoker=invoker, top_k=1)

        called_tools = [call["tool_name"] for call in invoker.calls]
        self.assertIn("get_industry_stocks", called_tools)
        self.assertNotIn("get_industry_pe", called_tools)
        self.assertNotIn("get_sector_fund_flow", called_tools)
        self.assertNotIn("get_macro_gdp", called_tools)
        self.assertNotIn("search_news", called_tools)
        self.assertNotIn(OPEN_WEBSEARCH_SEARCH_TOOL, called_tools)
        self.assertEqual(research["web"], [])

    def test_fast_mode_skips_company_web_evidence(self) -> None:
        invoker = FakeMCPInvoker()
        tool_logs: list[dict[str, object]] = []

        with patch.dict("os.environ", {"POLICYCHAIN_MCP_FAST_MODE": "1"}):
            evidence = collect_company_web_evidence(
                [{"company_name": "\u793a\u4f8b\u79d1\u6280", "stock_code": "300001"}],
                invoker=invoker,
                tool_logs=tool_logs,
            )

        self.assertEqual(evidence, [])
        self.assertEqual(invoker.calls, [])
        self.assertTrue(any(log["tool_name"] == "company_web_evidence" and log["status"] == "skipped" for log in tool_logs))

    def test_collect_company_candidates_respects_cloud_candidate_budget(self) -> None:
        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [
                    {"\u540d\u79f0": "\u8f6f\u4ef6\u5f00\u53d1"},
                ],
                (CNFINANCIAL_SERVER, "get_concept_list"): [],
                (CNFINANCIAL_SERVER, "get_industry_stocks"): [
                    {"\u540d\u79f0": "\u516c\u53f8A", "\u4ee3\u7801": "300001"},
                    {"\u540d\u79f0": "\u516c\u53f8B", "\u4ee3\u7801": "300002"},
                    {"\u540d\u79f0": "\u516c\u53f8C", "\u4ee3\u7801": "300003"},
                ],
                (CNFINANCIAL_SERVER, "get_company_profile"): [],
            }
        )

        with patch.dict(
            "os.environ",
            {
                "POLICYCHAIN_MCP_MAX_COMPANY_CANDIDATES": "2",
                "POLICYCHAIN_MCP_COMPANY_ENRICH_TOOLS": "get_company_profile",
            },
        ):
            candidates = collect_company_candidates(
                [_ai_impact()],
                invoker=invoker,
                top_k_per_industry=3,
            )

        self.assertEqual(len(candidates), 2)
        self.assertEqual([candidate["stock_code"] for candidate in candidates], ["300001", "300002"])

    def test_company_enrichment_uses_profile_only_by_default(self) -> None:
        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [
                    {"\u540d\u79f0": "\u8f6f\u4ef6\u5f00\u53d1"},
                ],
                (CNFINANCIAL_SERVER, "get_concept_list"): [],
                (CNFINANCIAL_SERVER, "get_industry_stocks"): [
                    {"\u540d\u79f0": "\u793a\u4f8b\u79d1\u6280", "\u4ee3\u7801": "300001"},
                ],
                (CNFINANCIAL_SERVER, "get_company_profile"): [],
            }
        )

        collect_company_candidates([_ai_impact()], invoker=invoker, top_k_per_industry=1)

        called_tools = [call["tool_name"] for call in invoker.calls]
        self.assertIn("get_company_profile", called_tools)
        self.assertNotIn("get_segments_revenue", called_tools)
        self.assertNotIn("get_financial_indicators", called_tools)
        self.assertNotIn("get_competitors", called_tools)
        self.assertNotIn("get_company_announcements", called_tools)
        self.assertNotIn("get_stock_news", called_tools)

    def test_company_candidates_select_and_recall_each_impact_independently(self) -> None:
        def industry_stocks(server_name: str, tool_name: str, arguments: dict[str, object]) -> list[dict[str, object]]:
            industry = str(arguments.get("industry") or "")
            if industry == "汽车整车":
                return [
                    {"名称": "整车公司", "代码": "000001", "主营业务": "新能源汽车整车制造"},
                    {"名称": "共用公司", "代码": "000003", "主营业务": "新能源汽车与动力电池系统"},
                ]
            if industry == "动力电池":
                return [
                    {"名称": "电池公司", "代码": "300002", "主营业务": "动力电池制造"},
                    {"名称": "共用公司", "代码": "000003", "主营业务": "新能源汽车与动力电池系统"},
                ]
            return []

        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [{"名称": "汽车整车"}, {"名称": "动力电池"}],
                (CNFINANCIAL_SERVER, "get_concept_list"): [{"名称": "新能源汽车"}],
                (CNFINANCIAL_SERVER, "get_industry_stocks"): industry_stocks,
                (CNFINANCIAL_SERVER, "search_stock"): [
                    {"名称": "共用公司", "代码": "000003", "主营业务": "新能源汽车与动力电池系统"},
                ],
                (CNFINANCIAL_SERVER, "get_company_profile"): [],
            }
        )
        tool_logs: list[dict[str, object]] = []

        with patch.dict("os.environ", {"POLICYCHAIN_MCP_MAX_COMPANY_CANDIDATES": "5"}):
            candidates = collect_company_candidates(
                [_vehicle_impact(), _battery_impact()],
                invoker=invoker,
                top_k_per_industry=3,
                tool_logs=tool_logs,
            )

        selected = [log for log in tool_logs if log["tool_name"] == "select_sectors"]
        self.assertEqual([log["impact_id"] for log in selected], ["IMP-001", "IMP-002"])
        self.assertEqual(selected[0]["selected_industries"][0]["name"], "汽车整车")
        self.assertEqual(selected[1]["selected_industries"][0]["name"], "动力电池")
        industry_calls = [call["arguments"]["industry"] for call in invoker.calls if call["tool_name"] == "get_industry_stocks"]
        self.assertIn("汽车整车", industry_calls)
        self.assertIn("动力电池", industry_calls)
        search_calls = [call for call in invoker.calls if call["tool_name"] == "search_stock"]
        self.assertGreaterEqual(len(search_calls), 2)
        self.assertEqual({item["stock_code"] for item in candidates}, {"000001", "300002", "000003"})
        summaries = [log for log in tool_logs if log["tool_name"] == "company_candidate_pipeline"]
        self.assertEqual({log["impact_id"] for log in summaries}, {"IMP-001", "IMP-002"})
        self.assertTrue(all("raw_count" in log and "dedup_count" in log and "truncated_count" in log for log in summaries))

    def test_duplicate_company_is_enriched_once_and_merges_full_provenance(self) -> None:
        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [{"名称": "汽车整车"}, {"名称": "动力电池"}],
                (CNFINANCIAL_SERVER, "get_concept_list"): [],
                (CNFINANCIAL_SERVER, "get_industry_stocks"): [
                    {"名称": "共用公司", "代码": "000003", "主营业务": "新能源汽车与动力电池系统"},
                ],
                (CNFINANCIAL_SERVER, "search_stock"): [
                    {"名称": "共用公司", "代码": "000003", "主营业务": "新能源汽车与动力电池系统"},
                ],
                (CNFINANCIAL_SERVER, "get_company_profile"): [{"主营业务": "整车与动力电池系统"}],
            }
        )

        candidates = collect_company_candidates(
            [_vehicle_impact(), _battery_impact()],
            invoker=invoker,
            top_k_per_industry=3,
        )

        self.assertEqual(len(candidates), 1)
        profile_calls = [call for call in invoker.calls if call["tool_name"] == "get_company_profile"]
        self.assertEqual(len(profile_calls), 1)
        provenance = candidates[0]["provenance"]
        self.assertEqual({item["impact_id"] for item in provenance}, {"IMP-001", "IMP-002"})
        self.assertTrue(all(item["tool_call_id"] for item in provenance))
        self.assertTrue({item["tool"] for item in provenance} >= {"get_industry_stocks", "search_stock"})

    def test_sector_catalogs_are_internal_only_not_research_evidence(self) -> None:
        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [{"名称": "半导体"}, {"名称": "动力电池"}],
                (CNFINANCIAL_SERVER, "get_concept_list"): [{"名称": "白酒"}],
                (CNFINANCIAL_SERVER, "get_industry_stocks"): [],
            }
        )

        with patch.dict("os.environ", {"POLICYCHAIN_MCP_FAST_MODE": "1"}):
            research = collect_impact_research({}, [_battery_impact()], invoker=invoker, top_k=1)

        self.assertFalse(any(item["tool_name"] in {"get_industry_list", "get_concept_list"} for item in research["cnfinancial"]))
        catalog_logs = [log for log in research["tool_logs"] if log["tool_name"] in {"get_industry_list", "get_concept_list"}]
        self.assertTrue(catalog_logs)
        self.assertTrue(all(log.get("internal_only") is True for log in catalog_logs))
        self.assertNotIn("半导体", str(research["cnfinancial"]))
        self.assertNotIn("白酒", str(research["cnfinancial"]))

    def test_only_valid_cnfinancial_react_company_is_promoted_and_path_bound(self) -> None:
        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_company_profile"): [{"主营业务": "动力电池制造与储能系统"}],
            }
        )
        evidence = [
            {
                "server_name": CNFINANCIAL_SERVER,
                "tool_name": "search_stock",
                "impact_id": "IMP-001",
                "query": "动力电池",
                "react_step": 1,
                "source_type": "cnfinancial_react",
                "raw_payload": {"名称": "电池公司", "代码": "300002", "主营业务": "动力电池制造"},
            },
            {
                "server_name": OPEN_WEBSEARCH_SERVER,
                "tool_name": "search",
                "impact_id": "IMP-001",
                "title": "网页提到公司",
                "raw_payload": {"名称": "网页提到公司", "代码": "300099"},
            },
            {
                "server_name": CNFINANCIAL_SERVER,
                "tool_name": "search_stock",
                "impact_id": "IMP-001",
                "react_step": 2,
                "raw_payload": {"名称": "缺代码公司"},
            },
        ]

        candidates, audit = merge_react_company_candidates(
            [],
            [_battery_impact()],
            evidence,
            invoker=invoker,
        )

        self.assertEqual([item["company_name"] for item in candidates], ["电池公司"])
        self.assertEqual(candidates[0]["impact_ids"], ["IMP-001"])
        self.assertEqual(candidates[0]["provenance"][0]["react_step"], 1)
        self.assertEqual(candidates[0]["provenance"][0]["source_type"], "cnfinancial_react")
        self.assertEqual([item["decision"] for item in audit], ["accept", "reject", "reject"])
        self.assertIn("Web", audit[1]["reason"])
        self.assertEqual(len([call for call in invoker.calls if call["tool_name"] == "get_company_profile"]), 1)


def _ai_impact() -> dict[str, object]:
    return {
        "industry": "\u6a21\u578b\u5b89\u5168\u8bc4\u4f30\u670d\u52a1",
        "chain_segment": "\u6a21\u578b\u5b89\u5168\u8bc4\u4f30\u670d\u52a1",
        "transmission_logic": "\u751f\u6210\u5f0f\u4eba\u5de5\u667a\u80fd\u670d\u52a1\u9700\u8981\u7b97\u6cd5\u6a21\u578b\u5b89\u5168\u8bc4\u4f30\u548c\u5408\u89c4\u5907\u6848",
        "business_variables": ["\u5b89\u5168\u8bc4\u4f30\u9700\u6c42", "\u5408\u89c4\u6210\u672c"],
        "affected_company_types": ["\u6a21\u578b\u8bc4\u6d4b\u673a\u6784", "\u4eba\u5de5\u667a\u80fd\u8f6f\u4ef6\u670d\u52a1\u5546"],
        "conditions": ["\u9700\u516c\u544a\u548c\u5b98\u7f51\u9a8c\u8bc1\u4e3b\u8425\u4e1a\u52a1"],
        "risks": [],
    }


def _vehicle_impact() -> dict[str, object]:
    return {
        "industry": "汽车整车制造",
        "chain_segment": "新能源汽车整车",
        "transmission_logic": "新能源汽车推广带动汽车整车制造需求",
        "business_variables": ["整车销量"],
        "affected_company_types": ["新能源汽车整车制造商"],
        "conditions": [],
        "risks": [],
    }


def _battery_impact() -> dict[str, object]:
    return {
        "industry": "动力电池",
        "chain_segment": "动力电池制造",
        "transmission_logic": "新能源汽车推广带动动力电池装机需求",
        "business_variables": ["电池装机量"],
        "affected_company_types": ["动力电池制造商"],
        "conditions": [],
        "risks": [],
    }


if __name__ == "__main__":
    unittest.main()
