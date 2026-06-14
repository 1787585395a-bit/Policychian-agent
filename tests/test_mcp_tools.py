from __future__ import annotations

import unittest

from policychain.mcp import FakeMCPInvoker
from policychain.tools.mcp_tools import (
    CNINFO_DOWNLOAD_ANNUAL_REPORTS_TOOL,
    CNINFO_QUERY_ANNUAL_REPORTS_TOOL,
    CNFINANCIAL_SERVER,
    OPEN_WEBSEARCH_SEARCH_TOOL,
    CNINFO_SERVER,
    OPEN_WEBSEARCH_SERVER,
    collect_company_candidates,
    collect_impact_research,
    fetch_web_content,
    search_web,
    select_recent_annual_reports,
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

    def test_select_recent_annual_reports_uses_disclosed_report_years(self) -> None:
        selected = select_recent_annual_reports(
            {
                "reports": [
                    {"title": "2022 年年度报告"},
                    {"title": "2025 年年度报告"},
                    {"title": "2024 年年度报告"},
                ]
            }
        )

        self.assertEqual([item["title"] for item in selected], ["2025 年年度报告", "2024 年年度报告"])

    def test_select_recent_annual_reports_filters_cninfo_stock_code_and_summary(self) -> None:
        selected = select_recent_annual_reports(
            {
                "reports": [
                    {"announcementTitle": "2025年年度报告摘要", "secCode": "000888"},
                    {"announcementTitle": "2025年年度报告", "secCode": "000888"},
                    {"announcementTitle": "2024年年度报告", "secCode": "000888"},
                    {"announcementTitle": "2025年年度报告", "secCode": "600540"},
                ]
            },
            stock_code="000888",
        )

        self.assertEqual([item["announcementTitle"] for item in selected], ["2025年年度报告", "2024年年度报告"])

    def test_fake_invoker_records_cninfo_calls(self) -> None:
        invoker = FakeMCPInvoker(
            {
                (CNINFO_SERVER, CNINFO_QUERY_ANNUAL_REPORTS_TOOL): [{"year": 2025}],
                (CNINFO_SERVER, CNINFO_DOWNLOAD_ANNUAL_REPORTS_TOOL): [{"content": "主营业务包括模型安全评估服务"}],
            }
        )

        invoker.invoke(CNINFO_SERVER, CNINFO_QUERY_ANNUAL_REPORTS_TOOL, {"stock_code": "300001"})
        invoker.invoke(CNINFO_SERVER, CNINFO_DOWNLOAD_ANNUAL_REPORTS_TOOL, {"stock_code": "300001", "year": 2025})

        self.assertEqual([call["tool_name"] for call in invoker.calls], [CNINFO_QUERY_ANNUAL_REPORTS_TOOL, CNINFO_DOWNLOAD_ANNUAL_REPORTS_TOOL])


    def test_collect_impact_research_uses_cnfinancial_official_arguments(self) -> None:
        invoker = FakeMCPInvoker(
            {
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

        collect_impact_research({}, [_ai_impact()], invoker=invoker, top_k=1)

        industry_call = next(call for call in invoker.calls if call["tool_name"] == "get_industry_stocks")
        news_call = next(call for call in invoker.calls if call["tool_name"] == "search_news")
        fund_flow_call = next(call for call in invoker.calls if call["tool_name"] == "get_sector_fund_flow")

        self.assertEqual(set(industry_call["arguments"]), {"industry"})
        self.assertIn(
            industry_call["arguments"]["industry"],
            {"\u8f6f\u4ef6\u5f00\u53d1", "\u4e92\u8054\u7f51\u670d\u52a1", "\u8ba1\u7b97\u673a\u8bbe\u5907"},
        )
        self.assertEqual(set(news_call["arguments"]), {"keyword", "num_results"})
        self.assertEqual(fund_flow_call["arguments"], {})

    def test_collect_company_candidates_prefers_industry_stocks_and_enriches_by_symbol(self) -> None:
        invoker = FakeMCPInvoker(
            {
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

        candidates = collect_company_candidates([_ai_impact()], invoker=invoker, top_k_per_industry=1)

        self.assertEqual(candidates[0]["company_name"], "\u793a\u4f8b\u79d1\u6280")
        self.assertEqual(candidates[0]["stock_code"], "300001")
        self.assertEqual(candidates[0]["revenue_relevance"], "28%")
        industry_call = next(call for call in invoker.calls if call["tool_name"] == "get_industry_stocks")
        profile_call = next(call for call in invoker.calls if call["tool_name"] == "get_company_profile")
        self.assertEqual(set(industry_call["arguments"]), {"industry"})
        self.assertEqual(profile_call["arguments"], {"symbol": "300001"})


def _ai_impact() -> dict[str, object]:
    return {
        "industry": "\u6a21\u578b\u5b89\u5168\u8bc4\u4f30\u670d\u52a1",
        "chain_segment": "\u6a21\u578b\u5b89\u5168\u8bc4\u4f30\u670d\u52a1",
        "transmission_logic": "\u751f\u6210\u5f0f\u4eba\u5de5\u667a\u80fd\u670d\u52a1\u9700\u8981\u7b97\u6cd5\u6a21\u578b\u5b89\u5168\u8bc4\u4f30\u548c\u5408\u89c4\u5907\u6848",
        "business_variables": ["\u5b89\u5168\u8bc4\u4f30\u9700\u6c42", "\u5408\u89c4\u6210\u672c"],
        "affected_company_types": ["\u6a21\u578b\u8bc4\u6d4b\u673a\u6784", "\u4eba\u5de5\u667a\u80fd\u8f6f\u4ef6\u670d\u52a1\u5546"],
        "conditions": ["\u9700\u5e74\u62a5\u9a8c\u8bc1\u4e3b\u8425\u4e1a\u52a1"],
        "risks": [],
    }


if __name__ == "__main__":
    unittest.main()
