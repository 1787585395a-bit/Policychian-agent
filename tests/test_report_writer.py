from __future__ import annotations

import unittest

from policychain.agents import (
    run_company_matcher,
    run_impact_analyst,
    run_policy_analyst,
    write_llm_research_report,
    write_research_report,
)
from policychain.state import PolicyResearchState
from tests.helpers import build_sample_store


PROHIBITED_TERMS = ("买入", "卖出", "目标价", "推荐股票")
REQUIRED_SECTIONS = (
    "PolicyChain 政策研究报告",
    "研究摘要",
    "主政策解读",
    "相似政策对比",
    "实施路径与行业影响",
    "A 股公司业务匹配",
    "不确定性与使用边界",
    "参考资料与工具依据",
)


class ReportWriterTests(unittest.TestCase):
    def test_write_research_report_contains_narrative_sections(self) -> None:
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
            self.assertNotIn("关键证据与引用", report)
            self.assertNotIn("外部证据与 MCP 工具结果", report)
            self.assertNotIn("ReAct 检索日志摘要", report)
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

    def test_reference_appendix_limits_evidence_and_tool_logs(self) -> None:
        state = PolicyResearchState(user_query="测试")
        state.evidence = [
            {"policy_id": f"POL-{index}", "chunk_id": f"C{index}", "text": f"本地证据 {index}"}
            for index in range(5)
        ]
        state.external_evidence = [
            {
                "server_name": "web-search",
                "tool_name": "search",
                "title": f"外部资料 {index}",
                "source_url": f"https://example.test/{index}",
            }
            for index in range(5)
        ]
        state.tool_call_logs = [
            {
                "server_name": "cn-financial",
                "tool_name": f"tool_{index}",
                "arguments": {"q": index},
                "status": "ok",
                "count": index,
                "error": "",
            }
            for index in range(7)
        ]

        report = write_research_report(state)

        self.assertIn("参考资料与工具依据", report)
        self.assertIn("POL-0", report)
        self.assertIn("POL-1", report)
        self.assertNotIn("POL-2", report)
        self.assertIn("外部资料 1", report)
        self.assertNotIn("外部资料 2", report)
        self.assertIn("cn-financial.tool_3", report)
        self.assertNotIn("cn-financial.tool_4", report)

    def test_report_covers_every_industry_path_and_no_match_reason(self) -> None:
        state = PolicyResearchState(user_query="测试")
        state.industry_impacts = [
            {
                "industry": "传统钢铁行业",
                "policy_measure": "推动数智化绿色化改造",
                "implementation_action": "开展节能降碳技改",
                "chain_segment": "钢铁冶炼与轧钢",
                "business_variables": ["技改资本开支", "环保合规成本"],
                "affected_company_types": ["钢铁冶炼企业"],
                "transmission_logic": "政策要求传统产业升级改造。",
                "impact_type": "direct",
                "direction": "mixed",
            },
            {
                "industry": "数智赋能产业链",
                "policy_measure": "推动数智技术赋能",
                "implementation_action": "建设算力和工业互联网平台",
                "chain_segment": "算力基础设施",
                "business_variables": ["算力需求", "平台渗透率"],
                "affected_company_types": ["数据中心与工业互联网服务商"],
                "transmission_logic": "各行业数字化改造拉动基础设施需求。",
                "impact_type": "indirect",
                "direction": "positive",
            },
            {
                "industry": "未来产业公司",
                "policy_measure": "前瞻布局未来产业",
                "implementation_action": "组织示范项目和研发攻关",
                "chain_segment": "未来产业研发验证",
                "business_variables": ["研发投入", "示范项目数量"],
                "affected_company_types": ["处于商业化早期的研发型企业"],
                "transmission_logic": "政策资源引导早期技术验证。",
                "impact_type": "potential",
                "direction": "positive",
            },
        ]
        state.company_coverage = [
            {"impact_id": "IMP-001", "industry": "传统钢铁行业", "passed_count": 0, "no_match_reason": "缺少业务证据。"},
            {"impact_id": "IMP-002", "industry": "数智赋能产业链", "passed_count": 0, "no_match_reason": "CNFinancial 未返回候选。"},
            {"impact_id": "IMP-003", "industry": "未来产业公司", "passed_count": 0, "no_match_reason": "路径过宽，暂无法绑定公司。"},
        ]

        report = write_research_report(state)

        self.assertIn("传统钢铁行业", report)
        self.assertIn("数智赋能产业链", report)
        self.assertIn("未来产业公司", report)
        self.assertIn("短期", report)
        self.assertIn("中长期", report)
        self.assertEqual(report.count("暂未形成可靠 A 股公司匹配"), 3)

    def test_write_research_report_explains_similar_policy_comparison_dimensions(self) -> None:
        state = PolicyResearchState(user_query="测试")
        state.similar_policy_matches = [
            {
                "policy_id": "POL-2024-LOCAL-0001",
                "title": "某省人工智能产业发展资金支持办法",
                "agency": "某省人民政府",
                "publish_date": "2024-01-01",
                "matched_text": "支持人工智能企业申报补贴和示范试点。",
                "source_url": "https://example.test/local-policy",
            }
        ]

        report = write_research_report(state)

        self.assertIn("层级/主体", report)
        self.assertIn("政策工具", report)
        self.assertIn("力度特征", report)
        self.assertIn("https://example.test/local-policy", report)

    def test_write_llm_research_report_uses_llm_body_and_compact_references(self) -> None:
        state = PolicyResearchState(user_query="测试")
        state.policy_analysis = {
            "policy_identity": {"policy_id": "INPUT-12345678", "title": "测试政策"},
            "policy_measures": ["推动人工智能安全评估"],
        }
        state.industry_impacts = [{"industry": "模型评估服务", "transmission_logic": "政策要求带来评估需求"}]
        state.company_matches = [{"company_name": "示例科技", "stock_code": "300001"}]
        state.external_evidence = [
            {"server_name": "cn-financial", "tool_name": "get_industry_list", "title": "行业列表"}
        ]

        client = _OneShotClient("# 自由报告\n\n模型自行组织的政策研究说明。")
        report = write_llm_research_report(state, client)

        self.assertEqual(len(client.calls), 1)
        self.assertIn("模型自行组织的政策研究说明", report)
        self.assertIn("参考资料与工具依据", report)
        self.assertNotIn("cn-financial.get_industry_list", report)
        self.assertNotIn("行业列表", client.calls[0][1])
        self.assertEqual(state.final_report, report)


class _OneShotClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


if __name__ == "__main__":
    unittest.main()
