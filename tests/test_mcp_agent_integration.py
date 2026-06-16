from __future__ import annotations

import json
import unittest

from policychain.agents import run_llm_policy_analyst, run_policy_analyst, run_impact_analyst
from policychain.agents.company_matcher import match_companies_for_impacts
from policychain.mcp import FakeMCPInvoker
from policychain.tools.mcp_tools import (
    CNFINANCIAL_SERVER,
    OPEN_WEBSEARCH_SEARCH_TOOL,
    OPEN_WEBSEARCH_SERVER,
)
from policychain.state import PolicyResearchState
from tests.helpers import build_sample_store


class RecordingLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


class MCPAgentIntegrationTests(unittest.TestCase):
    def test_policy_analyst_records_web_evidence_without_replacing_local_policy_identity(self) -> None:
        store = build_sample_store()
        invoker = _fake_policy_web_invoker()
        try:
            state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
            output = run_policy_analyst(state, store, mcp_invoker=invoker)

            self.assertEqual(output.policy_identity["policy_id"], "POL-2023-NAT-0048")
            self.assertTrue(state.policy_web_evidence)
            self.assertTrue(any(item["title"] == "官方解读" for item in state.external_evidence))
        finally:
            store.close()

    def test_llm_policy_analyst_prompt_receives_web_evidence(self) -> None:
        store = build_sample_store()
        invoker = _fake_policy_web_invoker()
        client = RecordingLLMClient(json.dumps(_policy_payload(), ensure_ascii=False))
        try:
            state = PolicyResearchState(user_query="生成式人工智能")
            run_llm_policy_analyst(state, store, llm_client=client, mcp_invoker=invoker)

            self.assertIn("官方解读", client.calls[-1][1])
            self.assertTrue(state.policy_web_evidence)
            self.assertTrue(state.react_traces)
        finally:
            store.close()

    def test_impact_analyst_records_cnfinancial_and_web_research(self) -> None:
        store = build_sample_store()
        invoker = _fake_industry_invoker()
        try:
            state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
            run_policy_analyst(state, store)
            run_impact_analyst(state, mcp_invoker=invoker)

            self.assertTrue(state.industry_research)
            self.assertTrue(any(item["tool_name"] == "get_industry_list" for item in state.industry_research))
            impact = state.industry_impacts[0]
            self.assertTrue(impact["chain_segment"])
            self.assertTrue(impact["business_variables"])
            self.assertTrue(impact["affected_company_types"])
        finally:
            store.close()

    def test_company_matcher_uses_cnfinancial_candidates_without_annual_reports(self) -> None:
        invoker = _fake_company_invoker(content="公司主营业务包括模型安全评估服务，相关产品形成分部收入。")
        output = match_companies_for_impacts([_industry_impact()], mcp_invoker=invoker)
        company = output.to_dict()["companies"][0]

        self.assertEqual(company["company_name"], "示例科技")
        self.assertEqual(company["stock_code"], "300001")
        self.assertNotIn("annual_report_evidence", company)
        self.assertTrue(company["business_evidence"])
        self.assertNotEqual(company["match_level"], "low")

    def test_company_matcher_does_not_downgrade_only_because_annual_reports_are_absent(self) -> None:
        invoker = _fake_company_invoker(content="本年度公司办公楼修缮完成。")
        output = match_companies_for_impacts([_industry_impact()], mcp_invoker=invoker)
        company = output.to_dict()["companies"][0]

        self.assertNotIn("annual_report_evidence", company)
        self.assertNotIn("未在最近两期年报中找到充分证据", str(company))
        self.assertGreater(company["confidence"], 0.45)


def _fake_policy_web_invoker() -> FakeMCPInvoker:
    return FakeMCPInvoker(
        {
            (OPEN_WEBSEARCH_SERVER, OPEN_WEBSEARCH_SEARCH_TOOL): [
                {
                    "title": "官方解读",
                    "source": "国家网信办",
                    "date": "2023-07-13",
                    "url": "https://example.test/interpretation",
                    "description": "生成式人工智能服务管理办法官方解读",
                }
            ]
        }
    )


def _fake_industry_invoker() -> FakeMCPInvoker:
    return FakeMCPInvoker(
        {
            (OPEN_WEBSEARCH_SERVER, OPEN_WEBSEARCH_SEARCH_TOOL): [
                {
                    "title": "行业数据",
                    "source": "行业协会",
                    "date": "2026-01-01",
                    "url": "https://example.test/industry",
                    "description": "产业规模和技术路线数据",
                }
            ],
            (CNFINANCIAL_SERVER, "get_industry_list"): [
                {
                    "名称": "软件开发",
                    "source": "CNFinancial",
                },
                {
                    "title": "软件服务行业",
                    "source": "CNFinancial",
                    "description": "行业板块列表",
                }
            ],
            (CNFINANCIAL_SERVER, "get_concept_list"): [
                {"名称": "人工智能", "source": "CNFinancial"}
            ],
            (CNFINANCIAL_SERVER, "search_news"): [
                {
                    "title": "AI 合规行业新闻",
                    "source": "CNFinancial",
                    "description": "行业新闻摘要",
                }
            ],
        }
    )


def _fake_company_invoker(content: str) -> FakeMCPInvoker:
    return FakeMCPInvoker(
        {
            (CNFINANCIAL_SERVER, "get_industry_list"): [
                {"名称": "软件开发"},
            ],
            (CNFINANCIAL_SERVER, "get_concept_list"): [
                {"名称": "人工智能"},
            ],
            (CNFINANCIAL_SERVER, "get_industry_stocks"): [
                {
                    "company_name": "示例科技",
                    "stock_code": "300001",
                    "main_business": "模型安全评估服务",
                    "description": "提供模型安全评估和合规审计",
                }
            ],
            (CNFINANCIAL_SERVER, "search_stock"): [
                {
                    "company_name": "示例科技",
                    "stock_code": "300001",
                    "main_business": "模型安全评估服务",
                    "description": "提供模型安全评估和合规审计",
                }
            ],
            (CNFINANCIAL_SERVER, "get_company_profile"): [
                {
                    "main_business": "模型安全评估服务",
                    "revenue_ratio": "35%",
                    "source": "CNFinancial",
                }
            ],
        }
    )


def _industry_impact() -> dict[str, object]:
    return {
        "industry": "模型安全评估服务",
        "chain_segment": "模型安全评估服务",
        "transmission_logic": "政策要求服务提供者开展安全评估，影响模型安全评估服务需求。",
        "business_variables": ["安全评估需求", "合规成本"],
        "affected_company_types": ["模型评测机构"],
        "conditions": ["需公告和官网验证主营业务"],
        "risks": [],
    }


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
            "reasons": ["包含明确义务要求"],
            "uncertainties": [],
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
        "uncertainties": [],
    }


if __name__ == "__main__":
    unittest.main()
