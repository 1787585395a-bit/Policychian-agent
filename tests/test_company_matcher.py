from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from policychain.agents import (
    match_companies_for_impacts,
    run_company_matcher,
    run_impact_analyst,
    run_policy_analyst,
)
from policychain.agents.company_matcher import (
    audit_company_match_output,
    match_candidate_records_to_impacts,
    resolve_company_match_limit,
)
from policychain.mcp import FakeMCPInvoker, MCPToolError
from policychain.schemas import CompanyMatch, CompanyMatchOutput
from policychain.state import PolicyResearchState
from policychain.tools.mcp_tools import CNFINANCIAL_SERVER
from tests.helpers import build_sample_store


PROHIBITED_TERMS = ("买入", "卖出", "目标价", "推荐股票")
MATCH_LEVELS = {"high", "medium", "low"}


class CompanyMatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self._prior_company_discovery_mode = os.environ.get("POLICYCHAIN_COMPANY_DISCOVERY_MODE")
        os.environ["POLICYCHAIN_COMPANY_DISCOVERY_MODE"] = "legacy_cnfinancial"

    def tearDown(self) -> None:
        if self._prior_company_discovery_mode is None:
            os.environ.pop("POLICYCHAIN_COMPANY_DISCOVERY_MODE", None)
        else:
            os.environ["POLICYCHAIN_COMPANY_DISCOVERY_MODE"] = self._prior_company_discovery_mode

    def test_match_companies_for_impacts_returns_structured_matches(self) -> None:
        impacts = [
            {
                "industry": "生成式人工智能服务",
                "impact_type": "direct",
                "direction": "mixed",
                "transmission_logic": "政策直接规范生成式人工智能服务提供、备案和投诉受理责任。",
                "conditions": ["需要结合监管执行口径判断。"],
                "risks": ["合规成本可能上升。"],
            }
        ]
        output = match_companies_for_impacts(impacts)
        result = output.to_dict()

        self.assertIsInstance(output, CompanyMatchOutput)
        self.assertEqual(result["companies"], [])
        self.assertTrue(result["uncertainties"])
        self.assertTrue(any("mock" in item for item in result["uncertainties"]))

    def test_run_company_matcher_writes_shared_state(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
            run_policy_analyst(state, store)
            run_impact_analyst(state)
            output = run_company_matcher(state)

            self.assertEqual(output.companies, [])
            self.assertEqual(len(state.company_candidates), len(state.company_matches))
            self.assertEqual(state.company_matches, [])
            self.assertTrue(state.uncertainties)
        finally:
            store.close()

    def test_company_matches_include_business_evidence_and_policy_link(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
            run_policy_analyst(state, store)
            run_impact_analyst(state)
            output = run_company_matcher(state)
            companies = output.to_dict()["companies"]

            self.assertEqual(companies, [])
            self.assertTrue(any("候选公司" in item for item in state.uncertainties))
        finally:
            store.close()

    def test_company_matcher_binds_companies_to_each_industry_path_and_audits_relevance(self) -> None:
        impacts = [
            {
                "industry": "传统钢铁行业",
                "impact_type": "direct",
                "direction": "mixed",
                "policy_measure": "推动数智化绿色化改造",
                "implementation_action": "开展节能降碳技改",
                "chain_segment": "钢铁冶炼与轧钢",
                "business_variables": ["技改资本开支", "环保合规成本"],
                "affected_company_types": ["钢铁冶炼企业"],
                "transmission_logic": "政策要求钢铁行业进行数智化和绿色化技术改造。",
                "conditions": [],
                "risks": [],
            },
            {
                "industry": "数智赋能产业链",
                "impact_type": "indirect",
                "direction": "positive",
                "policy_measure": "推动数智技术赋能",
                "implementation_action": "建设算力和工业互联网平台",
                "chain_segment": "算力基础设施",
                "business_variables": ["算力需求", "平台渗透率"],
                "affected_company_types": ["数据中心与工业互联网服务商"],
                "transmission_logic": "各行业数字化改造拉动算力、数据中心和工业互联网平台需求。",
                "conditions": [],
                "risks": [],
            },
            {
                "industry": "未来产业公司",
                "impact_type": "potential",
                "direction": "positive",
                "policy_measure": "前瞻布局未来产业",
                "implementation_action": "组织低空经济示范和研发攻关",
                "chain_segment": "低空经济研发验证",
                "business_variables": ["研发投入", "示范项目数量"],
                "affected_company_types": ["低空经济研发企业"],
                "transmission_logic": "政策资源引导未来产业验证。",
                "conditions": [],
                "risks": [],
            },
        ]
        invoker = _multi_path_company_invoker()

        output = match_companies_for_impacts(impacts, mcp_invoker=invoker, top_k_per_industry=3)
        companies = output.to_dict()["companies"]
        coverage = getattr(output, "_company_coverage")

        self.assertEqual(len(coverage), 3)
        self.assertTrue(any(item["impact_id"] == "IMP-001" and item["passed_count"] >= 1 for item in coverage))
        self.assertTrue(any(item["impact_id"] == "IMP-002" and item["passed_count"] >= 1 for item in coverage))
        self.assertTrue(any(item["impact_id"] == "IMP-003" and item["no_match_reason"] for item in coverage))
        self.assertTrue(any(company["company_name"] == "华菱钢铁" and company["impact_id"] == "IMP-001" for company in companies))
        self.assertTrue(any(company["company_name"] == "科华数据" and company["impact_id"] == "IMP-002" for company in companies))
        self.assertFalse(
            any(
                company["company_name"] == "泛化服务"
                and company["match_level"] in {"medium", "high"}
                for company in companies
            )
        )
        self.assertTrue(all(company["audit_reason"] for company in companies))

    def test_company_matcher_default_does_not_require_annual_report_verification(self) -> None:
        output = match_companies_for_impacts(
            [
                {
                    "industry": "生成式人工智能服务",
                    "impact_type": "direct",
                    "direction": "mixed",
                    "transmission_logic": "政策影响生成式人工智能服务运营和备案管理。",
                    "conditions": [],
                    "risks": [],
                }
            ]
        )

        result = output.to_dict()

        self.assertEqual(result["companies"], [])
        rendered = str(result)
        self.assertNotIn("年报", rendered)
        self.assertNotIn("CNINFO", rendered)

    def test_company_matcher_missing_impacts_returns_uncertainty(self) -> None:
        state = PolicyResearchState(user_query="生成式人工智能")
        output = run_company_matcher(state)
        result = output.to_dict()

        self.assertEqual(result["companies"], [])
        self.assertTrue(result["uncertainties"])
        self.assertEqual(state.company_matches, [])

    def test_company_matcher_does_not_emit_investment_advice_terms(self) -> None:
        store = build_sample_store()
        try:
            state = PolicyResearchState(user_query="生成式人工智能 公司影响")
            run_policy_analyst(state, store)
            run_impact_analyst(state)
            output = run_company_matcher(state)
            rendered = str(output.to_dict())

            for term in PROHIBITED_TERMS:
                self.assertNotIn(term, rendered)
        finally:
            store.close()

    def test_generic_service_manufacturing_and_power_terms_are_rejected(self) -> None:
        impact = {
            "industry": "电力服务行业",
            "chain_segment": "设备制造服务",
            "transmission_logic": "企业获得电力服务和制造服务",
            "business_variables": ["服务规模"],
            "affected_company_types": ["制造企业"],
        }
        candidate = {
            "company_name": "泛化业务公司",
            "stock_code": "300999",
            "matched_business": "提供电力服务、设备制造和企业综合服务",
            "business_evidence": "主营业务为电力服务与设备制造服务",
            "candidate_source_tool": "get_industry_stocks",
            "impact_ids": ["IMP-001"],
            "provenance": [
                {
                    "impact_id": "IMP-001",
                    "tool": "get_industry_stocks",
                    "source_type": "cnfinancial_recall",
                    "tool_call_id": "tool-generic",
                }
            ],
        }

        matches, coverage, audit = match_candidate_records_to_impacts([impact], [candidate])

        self.assertEqual(matches, [])
        self.assertEqual(audit[0]["decision"], "reject")
        self.assertIn("泛化词", audit[0]["reason"])
        self.assertEqual(coverage[0]["passed_count"], 0)

    def test_llm_match_cannot_rebind_candidate_outside_retrieval_provenance(self) -> None:
        impacts = [
            {
                "industry": "算力基础设施",
                "chain_segment": "液冷服务器",
                "transmission_logic": "数据中心建设拉动液冷服务器需求",
            },
            {
                "industry": "天然气供应",
                "chain_segment": "天然气管网服务",
                "transmission_logic": "管网建设影响天然气服务需求",
            },
        ]
        record = {
            "company_name": "路径二候选",
            "stock_code": "300777",
            "matched_business": "天然气管网运营服务",
            "business_evidence": "主营天然气管网运营服务",
            "impact_ids": ["IMP-002"],
            "provenance": [{"impact_id": "IMP-002", "tool": "search_stock"}],
        }
        llm_output = CompanyMatchOutput(
            companies=[
                CompanyMatch(
                    company_name="路径二候选",
                    stock_code="300777",
                    industry_segment="算力基础设施",
                    matched_business="天然气能源服务",
                    match_level="medium",
                    impact_id="IMP-001",
                    confidence=0.7,
                )
            ]
        )

        audited = audit_company_match_output(llm_output, impacts, [record])

        self.assertEqual(audited.companies, [])
        audit = getattr(audited, "_audit_logs")[0]
        self.assertEqual(audit["decision"], "reject")
        self.assertEqual(audit["reason_code"], "path_provenance_mismatch")
        self.assertIn("IMP-002", audit["reason"])

    def test_llm_match_without_candidate_provenance_keeps_legacy_path_compatibility(self) -> None:
        impact = {
            "industry": "算力基础设施",
            "chain_segment": "液冷服务器",
            "transmission_logic": "数据中心建设拉动液冷服务器需求",
        }
        record = {
            "company_name": "兼容候选",
            "stock_code": "300778",
            "matched_business": "液冷服务器和数据中心设备",
            "business_evidence": "主营液冷服务器和数据中心设备",
        }
        llm_output = CompanyMatchOutput(
            companies=[
                CompanyMatch(
                    company_name="兼容候选",
                    stock_code="300778",
                    industry_segment="算力基础设施",
                    matched_business="液冷服务器和数据中心设备",
                    match_level="medium",
                    impact_id="IMP-001",
                    confidence=0.7,
                )
            ]
        )

        audited = audit_company_match_output(llm_output, [impact], [record])

        self.assertEqual([item.company_name for item in audited.companies], ["兼容候选"])

    def test_energy_umbrella_and_generic_service_cannot_match_compute_path(self) -> None:
        impact = {
            "industry": "算力基础设施",
            "chain_segment": "数据中心能源系统",
            "transmission_logic": "数据中心建设影响能源与电力服务需求",
            "business_variables": ["单位算力能耗"],
            "affected_company_types": ["数据中心运营商"],
        }
        weak = {
            "company_name": "天然气服务候选",
            "stock_code": "300779",
            "matched_business": "天然气、新能源和电力综合服务",
            "business_evidence": "主营天然气能源服务和电力服务",
            "impact_ids": ["IMP-001"],
        }
        specific = {
            "company_name": "液冷设备候选",
            "stock_code": "300780",
            "matched_business": "液冷服务器和数据中心制冷设备",
            "business_evidence": "主营液冷服务器及数据中心制冷设备",
            "impact_ids": ["IMP-001"],
        }

        matches, _coverage, audit = match_candidate_records_to_impacts([impact], [weak, specific])

        self.assertEqual([item.company_name for item in matches], ["液冷设备候选"])
        weak_audit = next(item for item in audit if item["company_name"] == "天然气服务候选")
        self.assertEqual(weak_audit["decision"], "reject")
        self.assertIn("泛化词", weak_audit["reason"])

    def test_web_only_candidate_is_rejected_even_with_specific_business_text(self) -> None:
        impact = {
            "industry": "海水淡化设备",
            "chain_segment": "反渗透膜组件",
            "transmission_logic": "海水淡化项目增加反渗透膜组件需求",
            "business_variables": ["膜组件需求"],
            "affected_company_types": ["膜组件供应商"],
        }
        candidate = {
            "company_name": "网页公司",
            "stock_code": "300888",
            "matched_business": "生产反渗透膜组件",
            "business_evidence": "网页称公司生产海水淡化反渗透膜组件",
            "impact_ids": ["IMP-001"],
            "provenance": [
                {
                    "impact_id": "IMP-001",
                    "tool": "search",
                    "source_type": "web_search",
                    "tool_call_id": "tool-web",
                }
            ],
        }

        matches, _coverage, audit = match_candidate_records_to_impacts([impact], [candidate])

        self.assertEqual(matches, [])
        self.assertEqual(audit[0]["decision"], "reject")
        self.assertIn("Web", audit[0]["reason"])

    def test_coverage_distinguishes_query_error_from_successful_empty(self) -> None:
        impact = {
            "industry": "动力电池",
            "chain_segment": "动力电池制造",
            "transmission_logic": "动力电池装机需求增加",
            "business_variables": ["装机量"],
            "affected_company_types": ["动力电池制造商"],
        }

        def disconnected(**_kwargs):
            raise MCPToolError("RemoteDisconnected: remote end closed connection without response")

        error_invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [{"名称": "动力电池"}],
                (CNFINANCIAL_SERVER, "get_concept_list"): [],
                (CNFINANCIAL_SERVER, "get_industry_stocks"): disconnected,
                (CNFINANCIAL_SERVER, "search_stock"): [],
            }
        )
        empty_invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [],
                (CNFINANCIAL_SERVER, "get_concept_list"): [],
                (CNFINANCIAL_SERVER, "search_stock"): [],
            }
        )

        error_output = match_companies_for_impacts([impact], mcp_invoker=error_invoker)
        empty_output = match_companies_for_impacts([impact], mcp_invoker=empty_invoker)
        error_coverage = getattr(error_output, "_company_coverage")[0]
        empty_coverage = getattr(empty_output, "_company_coverage")[0]

        self.assertEqual(error_coverage["retrieval_status"], "error")
        self.assertIn("查询失败", error_coverage["no_match_reason"])
        self.assertEqual(empty_coverage["retrieval_status"], "empty")
        self.assertIn("真实返回空", empty_coverage["no_match_reason"])

    def test_company_match_limit_defaults_to_three_and_never_exceeds_four(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_company_match_limit(), 3)
            self.assertEqual(resolve_company_match_limit(9), 4)
        with patch.dict("os.environ", {"POLICYCHAIN_MAX_COMPANY_MATCHES_PER_IMPACT": "4"}):
            self.assertEqual(resolve_company_match_limit(), 4)
        with patch.dict("os.environ", {"POLICYCHAIN_MAX_COMPANY_MATCHES_PER_IMPACT": "9"}):
            self.assertEqual(resolve_company_match_limit(), 4)


def _multi_path_company_invoker() -> FakeMCPInvoker:
    def industry_stocks(server_name: str, tool_name: str, arguments: dict[str, object]) -> list[dict[str, object]]:
        industry = str(arguments.get("industry") or "")
        if "钢铁" in industry:
            return [
                {
                    "company_name": "华菱钢铁",
                    "stock_code": "000932",
                    "main_business": "钢铁冶炼、轧钢生产和销售",
                    "description": "公司主营钢铁冶炼和轧钢产品，推进节能环保技改。",
                }
            ]
        if "软件" in industry or "互联网" in industry:
            return [
                {
                    "company_name": "科华数据",
                    "stock_code": "002335",
                    "main_business": "数据中心、算力基础设施和智慧电能服务",
                    "description": "公司提供 IDC 数据中心、云计算基础设施和算力相关服务。",
                }
            ]
        return []

    def search_stock(server_name: str, tool_name: str, arguments: dict[str, object]) -> list[dict[str, object]]:
        return [
            {
                "company_name": "泛化服务",
                "stock_code": "300999",
                "main_business": "",
                "description": "综合咨询服务。",
            }
        ]

    def company_profile(server_name: str, tool_name: str, arguments: dict[str, object]) -> list[dict[str, object]]:
        symbol = str(arguments.get("symbol") or "")
        if symbol == "000932":
            return [{"main_business": "钢铁冶炼、轧钢和钢材销售", "revenue_ratio": "90%"}]
        if symbol == "002335":
            return [{"main_business": "数据中心、算力基础设施、智慧电能", "revenue_ratio": "60%"}]
        return []

    return FakeMCPInvoker(
        {
            (CNFINANCIAL_SERVER, "get_industry_list"): [{"名称": "钢铁"}, {"名称": "软件开发"}, {"名称": "互联网服务"}],
            (CNFINANCIAL_SERVER, "get_concept_list"): [{"名称": "人工智能"}, {"名称": "数据中心"}],
            (CNFINANCIAL_SERVER, "get_industry_stocks"): industry_stocks,
            (CNFINANCIAL_SERVER, "search_stock"): search_stock,
            (CNFINANCIAL_SERVER, "get_company_profile"): company_profile,
            (CNFINANCIAL_SERVER, "get_company_info"): [],
            (CNFINANCIAL_SERVER, "get_segments_revenue"): [],
            (CNFINANCIAL_SERVER, "get_financial_indicators"): [],
            (CNFINANCIAL_SERVER, "get_growth_rates"): [],
            (CNFINANCIAL_SERVER, "get_competitors"): [],
            (CNFINANCIAL_SERVER, "get_company_announcements"): [],
            (CNFINANCIAL_SERVER, "get_stock_news"): [],
        }
    )


if __name__ == "__main__":
    unittest.main()
