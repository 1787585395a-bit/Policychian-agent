from __future__ import annotations

import unittest

from policychain.agents import (
    run_company_matcher,
    run_impact_analyst,
    run_policy_analyst,
    write_research_report,
)
from policychain.state import PolicyResearchState
from tests.helpers import build_sample_store


PROHIBITED_TERMS = ("买入", "卖出", "目标价", "推荐股票")
REQUIRED_SECTIONS = (
    "政策基本信息",
    "政策核心内容",
    "实施路径分析",
    "行业影响分析",
    "公司业务匹配清单",
    "关键证据与引用",
    "不确定性和风险提示",
)


class ReportWriterTests(unittest.TestCase):
    def test_write_research_report_contains_required_sections(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
            run_policy_analyst(state, store)
            run_impact_analyst(state)
            run_company_matcher(state)
            report = write_research_report(state)

            for section in REQUIRED_SECTIONS:
                self.assertIn(section, report)
            self.assertIn("POL-2023-NAT-0048", report)
            self.assertIn("生成式人工智能服务管理暂行办法", report)
            self.assertEqual(state.final_report, report)
        finally:
            store.close()

    def test_write_research_report_does_not_emit_prohibited_terms(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="生成式人工智能 公司影响")
            run_policy_analyst(state, store)
            run_impact_analyst(state)
            run_company_matcher(state)
            report = write_research_report(state)

            for term in PROHIBITED_TERMS:
                self.assertNotIn(term, report)
        finally:
            store.close()

    def test_write_research_report_includes_external_mcp_evidence(self) -> None:
        state = PolicyResearchState(user_query="测试")
        state.external_evidence = [
            {
                "server_name": "cn-financial",
                "tool_name": "get_industry_list",
                "title": "行业列表",
                "summary": "行业数据摘要",
            },
            {
                "server_name": "web-search",
                "tool_name": "search",
                "title": "官方解读",
                "source_url": "https://example.test/policy",
            },
        ]

        report = write_research_report(state)

        self.assertIn("MCP", report)
        self.assertIn("cn-financial.get_industry_list", report)
        self.assertIn("web-search.search", report)


if __name__ == "__main__":
    unittest.main()
