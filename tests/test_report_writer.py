from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

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

    def test_llm_company_section_is_replaced_and_rejected_company_context_is_not_prompted(self) -> None:
        state = _seven_path_state()
        state.company_candidates = [{"company_name": "诱导候选", "stock_code": "300999"}]
        state.company_seeds = [
            {
                "seed_id": "seed-rejected",
                "impact_id": "IMP-001",
                "proposed_name": "未验证Seed公司",
                "proposed_stock_code": "300997",
                "status": "unverified",
            }
        ]
        state.company_seed_audit = [
            {
                "seed_id": "seed-rejected",
                "impact_id": "IMP-001",
                "company_name": "身份冲突公司",
                "stock_code": "300998",
                "status": "rejected",
                "reason_code": "name_code_conflict",
            }
        ]
        state.company_research = [
            {
                "server_name": "web-search",
                "tool_name": "search",
                "company_name": "研究候选",
                "title": "研究候选业务资料",
                "source_url": "https://example.test/rejected-company",
            }
        ]
        state.external_evidence = list(state.company_research)
        state.uncertainties = ["诱导候选未通过审计。", "政策执行进度仍待验证。"]
        state.tool_call_logs = [
            {
                "server_name": "cn-financial",
                "tool_name": "get_company_profile",
                "arguments": {"symbol": "300999"},
                "status": "ok",
                "count": 1,
            }
        ]
        client = _OneShotClient(
            "# 政策传导说明\n\n正文只解释政策与行业路径。\n\n"
            "## A 股公司业务匹配\n\n- 编造公司（300888）：示例候选。\n\n"
            "## 风险说明\n\n执行进度仍有不确定性。"
        )

        report = write_llm_research_report(state, client)
        prompt = client.calls[0][0] + client.calls[0][1]

        for forbidden in (
            "诱导候选",
            "研究候选",
            "300999",
            "未验证Seed公司",
            "300997",
            "身份冲突公司",
            "300998",
            "编造公司",
            "300888",
        ):
            self.assertNotIn(
                forbidden,
                prompt
                if forbidden in {"诱导候选", "研究候选", "300999", "未验证Seed公司", "300997", "身份冲突公司", "300998"}
                else report,
            )
        self.assertNotIn("编造公司", report)
        self.assertNotIn("300888", report)
        self.assertEqual(report.count("## A 股公司业务匹配"), 1)

        for index in range(1, 8):
            self.assertIn(f"IMP-{index:03d}", report)
        self.assertEqual(report.count("暂未形成可靠 A 股公司匹配"), 7)

    def test_unsafe_llm_report_falls_back_without_echoing_prohibited_or_fabricated_content(self) -> None:
        state = _seven_path_state()
        client = _OneShotClient(
            "# 报告\n\n## 相关公司\n编造公司（300888）。\n\n"
            "对于 投资者 而言，应、重点关注确定性，需求、利好和成长-叙事。"
        )
        recorder = Mock()

        with patch("policychain.agents.report_writer.current_run_recorder", return_value=recorder):
            report = write_llm_research_report(state, client)

        recorder.mark_fallback.assert_called_once()
        self.assertNotIn("编造公司", report)
        self.assertNotIn("300888", report)
        for forbidden in ("对于 投资者 而言", "应、重点关注", "确定性，需求", "利好", "成长-叙事"):
            self.assertNotIn(forbidden, report)
        for index in range(1, 8):
            self.assertIn(f"IMP-{index:03d}", report)

    def test_llm_report_company_appendix_contains_only_approved_matches_and_caps_each_path(self) -> None:
        state = PolicyResearchState(user_query="测试")
        state.industry_impacts = [
            {"industry": "路径一", "chain_segment": "液冷服务器", "business_variables": ["设备需求"]},
            {"industry": "路径二", "chain_segment": "反渗透膜", "business_variables": ["项目需求"]},
        ]
        state.company_matches = [
            _approved_match("白名单甲", "300001", "IMP-001", 0.92),
            _approved_match("白名单乙", "300002", "IMP-001", 0.85),
            _approved_match("白名单丙", "300003", "IMP-001", 0.75),
            _approved_match("白名单超额", "300004", "IMP-001", 0.65),
            _approved_match("白名单丁", "300005", "IMP-002", 0.8),
        ]
        state.company_coverage = [
            {"impact_id": "IMP-001", "industry": "路径一", "passed_count": 4},
            {"impact_id": "IMP-002", "industry": "路径二", "passed_count": 1},
        ]
        client = _OneShotClient(
            "# 政策说明\n\n行业传导正文。\n\n## 公司关注清单\n- 编造公司（300888）"
        )

        with patch.dict("os.environ", {"POLICYCHAIN_MAX_COMPANY_MATCHES_PER_IMPACT": ""}):
            report = write_llm_research_report(state, client)

        for name in ("白名单甲", "白名单乙", "白名单丙", "白名单丁"):
            self.assertIn(name, report)
        self.assertNotIn("白名单超额", report)
        self.assertNotIn("300004", report)
        self.assertNotIn("编造公司", report)
        self.assertNotIn("300888", report)
        self.assertEqual(report.count("## A 股公司业务匹配"), 1)

        with patch.dict("os.environ", {"POLICYCHAIN_MAX_COMPANY_MATCHES_PER_IMPACT": "4"}):
            report_with_four = write_llm_research_report(state, client)
        self.assertIn("白名单超额", report_with_four)
        self.assertIn("300004", report_with_four)


class _OneShotClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


def _seven_path_state() -> PolicyResearchState:
    state = PolicyResearchState(user_query="测试")
    state.industry_impacts = [
        {
            "industry": f"测试路径{index}",
            "chain_segment": f"测试产业链{index}",
            "business_variables": [f"变量{index}"],
        }
        for index in range(1, 8)
    ]
    state.company_coverage = [
        {
            "impact_id": f"IMP-{index:03d}",
            "industry": f"测试路径{index}",
            "passed_count": 0,
            "no_match_reason": f"路径{index}没有通过审计的公司。",
        }
        for index in range(1, 8)
    ]
    return state


def _approved_match(name: str, code: str, impact_id: str, confidence: float) -> dict[str, object]:
    return {
        "company_name": name,
        "stock_code": code,
        "impact_id": impact_id,
        "chain_segment": "具体设备",
        "matched_business": "具体设备主营业务",
        "business_evidence": [{"text": "公开资料显示主营具体设备"}],
        "negative_evidence": ["收入占比待核验"],
        "match_level": "medium",
        "confidence": confidence,
        "audit_reason": "已通过业务相关性审查",
    }


if __name__ == "__main__":
    unittest.main()
