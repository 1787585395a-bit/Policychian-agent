from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from policychain.agents import (
    LLMCompanyMatchError,
    run_company_matcher,
    run_impact_analyst,
    run_llm_company_matcher,
    run_policy_analyst,
)
from policychain.agents.llm_company_matcher import _generate_company_seeds, _render_company_prompt
from policychain.schemas.agent_outputs import CompanyMatchOutput
from policychain.mcp import FakeMCPInvoker, MCPToolError
from policychain.observability import RunRecorder
from policychain.tools.mcp_tools import CNFINANCIAL_SERVER
from policychain.state import PolicyResearchState
from policychain.structured_output import StructuredOutputError
from tests.helpers import build_sample_store


class RecordingLLMClient:
    def __init__(self, response: str | list[str]) -> None:
        self.responses = list(response) if isinstance(response, list) else [response]
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


class LLMCompanyMatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_discovery_mode = os.environ.get("POLICYCHAIN_COMPANY_DISCOVERY_MODE")
        os.environ["POLICYCHAIN_COMPANY_DISCOVERY_MODE"] = "legacy_cnfinancial"

    def tearDown(self) -> None:
        if self._previous_discovery_mode is None:
            os.environ.pop("POLICYCHAIN_COMPANY_DISCOVERY_MODE", None)
        else:
            os.environ["POLICYCHAIN_COMPANY_DISCOVERY_MODE"] = self._previous_discovery_mode

    def test_run_llm_company_matcher_writes_state_from_valid_json(self) -> None:
        store = build_sample_store()
        client = _company_llm_client()
        try:
            state = _state_with_industry_impacts(store)
            output = run_llm_company_matcher(state, llm_client=client, mcp_invoker=_fake_company_invoker())

            self.assertIsInstance(output, CompanyMatchOutput)
            self.assertEqual(output.companies[0].company_name, "清源模型安全科技")
            self.assertGreaterEqual(len(state.company_candidates), 1)
            self.assertEqual(state.company_matches[0]["company_name"], "清源模型安全科技")
            self.assertTrue(state.uncertainties)
            self.assertEqual(len(client.calls), len(state.industry_impacts) * 2 + 1)
        finally:
            store.close()

    def test_run_llm_company_matcher_prompt_contains_impacts_and_company_candidates(self) -> None:
        store = build_sample_store()
        client = _company_llm_client()
        try:
            state = _state_with_industry_impacts(store)
            run_llm_company_matcher(state, llm_client=client, mcp_invoker=_fake_company_invoker())

            system_prompt, user_prompt = client.calls[-1]
            self.assertIn("Company Matcher", system_prompt)
            self.assertIn("业务相关性匹配", system_prompt)
            self.assertIn("生成式人工智能服务", user_prompt)
            self.assertIn("清源模型安全科技", user_prompt)
            self.assertIn("只输出一个合法 JSON 对象", user_prompt)
        finally:
            store.close()

    def test_run_llm_company_matcher_missing_impacts_does_not_call_llm(self) -> None:
        client = RecordingLLMClient(json.dumps(_company_payload(), ensure_ascii=False))
        state = PolicyResearchState(user_query="生成式人工智能")

        output = run_llm_company_matcher(state, llm_client=client)

        self.assertEqual(output.companies, [])
        self.assertEqual(state.company_candidates, [])
        self.assertEqual(state.company_matches, [])
        self.assertEqual(client.calls, [])
        self.assertTrue(state.uncertainties)

    def test_run_llm_company_matcher_no_company_records_calls_seed_generator_per_impact(self) -> None:
        client = RecordingLLMClient(json.dumps(_seed_payload(), ensure_ascii=False))
        state = PolicyResearchState(
            user_query="无匹配行业",
            industry_impacts=[
                {
                    "impact_id": "IMP-001",
                    "industry": "无匹配行业",
                    "transmission_logic": "没有本地公司资料覆盖",
                    "conditions": [],
                    "risks": [],
                },
                {
                    "impact_id": "IMP-002",
                    "industry": "另一无匹配行业",
                    "transmission_logic": "没有可靠公司身份或业务证据",
                    "conditions": [],
                    "risks": [],
                },
            ],
        )

        output = run_llm_company_matcher(state, llm_client=client)

        self.assertEqual(output.companies, [])
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(all("seed 永远不等于 candidate" in system for system, _user in client.calls))
        self.assertEqual(state.company_candidates, [])
        self.assertTrue(state.uncertainties)

    def test_seed_generator_uses_unique_verified_count_and_remaining_deficit(self) -> None:
        impact = _seed_test_impact()
        for existing_count, expected_deficit in ((1, 3), (2, 2)):
            with self.subTest(existing_count=existing_count):
                records = [_verified_record(index) for index in range(1, existing_count + 1)]
                client = RecordingLLMClient(json.dumps(_seed_payload(), ensure_ascii=False))

                seeds, uncertainties = _generate_company_seeds(client, [impact], records, [])

                self.assertEqual(seeds, [])
                self.assertEqual(uncertainties, [])
                self.assertEqual(len(client.calls), 1)
                _system, user = client.calls[0]
                self.assertIn(f'"remaining_deficit":{expected_deficit}', user)
                for record in records:
                    self.assertIn(str(record["company_name"]), user)
                    self.assertIn(str(record["stock_code"]), user)

    def test_seed_generator_skips_only_after_four_unique_verified_identities(self) -> None:
        records = [_verified_record(index) for index in range(1, 5)]
        records.append(dict(records[0]))
        client = RecordingLLMClient(json.dumps(_seed_payload(), ensure_ascii=False))

        seeds, uncertainties = _generate_company_seeds(client, [_seed_test_impact()], records, [])

        self.assertEqual(seeds, [])
        self.assertEqual(uncertainties, [])
        self.assertEqual(client.calls, [])

    def test_multi_impact_react_failure_isolated_and_all_underfilled_paths_seeded(self) -> None:
        def web_search(*, arguments, **_kwargs):
            if arguments.get("query") == "path-two-evidence":
                return [
                    {
                        "title": "路径二公开业务资料",
                        "description": "路径二设备需求的公开资料。",
                        "url": "https://example.test/path-two",
                        "date": "2026-07-01",
                    }
                ]
            return []

        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [],
                (CNFINANCIAL_SERVER, "get_concept_list"): [],
                (CNFINANCIAL_SERVER, "search_stock"): [],
                (CNFINANCIAL_SERVER, "get_company_profile"): [
                    {"main_business": "路径一环保材料设备", "data_date": "2025"}
                ],
                (CNFINANCIAL_SERVER, "get_company_info"): [],
                ("web-search", "search"): web_search,
            }
        )
        impact_one = {
            **_seed_test_impact(),
            "impact_id": "IMP-001",
            "industry": "路径一环保材料",
        }
        impact_two = {
            **_seed_test_impact(),
            "impact_id": "IMP-002",
            "industry": "路径二环保设备",
        }
        unresolved_seed = _seed_payload(
            [
                {
                    "impact_id": "IMP-001",
                    "proposed_name": "未验证路径一公司",
                    "historical_names": [],
                    "proposed_stock_code": "300881",
                    "seed_reason": "可能提供路径一环保材料设备",
                    "origin_channels": ["llm"],
                }
            ]
        )
        client = RecordingLLMClient(
            [
                "invalid planner json",
                json.dumps(
                    {
                        "thought": "补充路径二公开资料",
                        "action": "web.search",
                        "arguments": {"query": "path-two-evidence", "top_k": 1},
                    },
                    ensure_ascii=False,
                ),
                _react_finish(),
                json.dumps(unresolved_seed, ensure_ascii=False),
                json.dumps(_seed_payload(), ensure_ascii=False),
            ]
        )
        state = PolicyResearchState(
            user_query="多路径环保政策",
            industry_impacts=[impact_one, impact_two],
        )

        output = run_llm_company_matcher(state, llm_client=client, mcp_invoker=invoker)

        self.assertEqual(output.companies, [])
        self.assertEqual(state.company_candidates, [])
        self.assertEqual(len(client.calls), 5)
        self.assertTrue(all("公司身份线索生成器" in client.calls[index][0] for index in (3, 4)))
        self.assertIn("IMP-001", client.calls[3][1])
        self.assertIn("IMP-002", client.calls[4][1])
        self.assertTrue(
            any(
                trace.get("impact_id") == "IMP-001" and trace.get("reason_code") == "planner_invalid_json"
                for trace in state.react_traces
            )
        )
        self.assertTrue(
            any(
                trace.get("impact_id") == "IMP-002" and trace.get("action") == "web.search"
                for trace in state.react_traces
            )
        )
        self.assertTrue(
            any(item.get("impact_id") == "IMP-002" and item.get("title") == "路径二公开业务资料" for item in state.company_research)
        )
        self.assertEqual(len(state.company_seed_audit), 1)
        self.assertEqual(state.company_seed_audit[0]["status"], "unresolved")
        self.assertNotIn("未验证路径一公司", str(state.company_candidates))

    def test_seed_pipeline_promotes_only_officially_verified_business_candidate(self) -> None:
        def search_stock(*, arguments, **_kwargs):
            if arguments.get("keyword") == "虚构膜科技":
                return [{"company_name": "虚构膜科技", "stock_code": "300123"}]
            return []

        impact = {
            "impact_id": "IMP-001",
            "industry": "海水淡化设备",
            "chain_segment": "环保反渗透膜材料",
            "transmission_logic": "示范项目采购带动环保反渗透膜材料需求",
            "business_variables": ["环保膜材料需求"],
            "affected_company_types": ["海水淡化设备供应商"],
            "conditions": [],
            "risks": [],
        }
        seed_payload = _seed_payload(
            [
                {
                    "impact_id": "IMP-001",
                    "proposed_name": "虚构膜科技",
                    "historical_names": [],
                    "proposed_stock_code": "300123",
                    "seed_reason": "可能提供反渗透膜组件",
                    "origin_channels": ["llm"],
                }
            ]
        )
        company_payload = _company_payload_for("虚构膜科技", "300123", "IMP-001")
        client = RecordingLLMClient(
            [
                _react_finish(),
                json.dumps(seed_payload, ensure_ascii=False),
                json.dumps(company_payload, ensure_ascii=False),
            ]
        )
        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [],
                (CNFINANCIAL_SERVER, "get_concept_list"): [],
                (CNFINANCIAL_SERVER, "search_stock"): search_stock,
                (CNFINANCIAL_SERVER, "get_company_profile"): [
                    {"main_business": "环保反渗透膜材料与海水淡化设备", "data_date": "2025"}
                ],
                (CNFINANCIAL_SERVER, "get_company_info"): [
                    {
                        "company_name": "虚构膜科技",
                        "stock_code": "300123",
                        "listing_status": "active",
                        "data_date": "2026-06-01",
                    }
                ],
                ("web-search", "search"): [
                    {
                        "title": "虚构膜科技主营业务",
                        "description": "公司主营环保反渗透膜材料与海水淡化设备。",
                        "url": "https://www.cninfo.com.cn/fake/300123",
                        "date": "2026-06-01",
                    }
                ],
            }
        )
        state = PolicyResearchState(user_query="海水淡化政策", industry_impacts=[impact])

        output = run_llm_company_matcher(state, llm_client=client, mcp_invoker=invoker)

        self.assertEqual([company.company_name for company in output.companies], ["虚构膜科技"])
        self.assertEqual(len(state.company_candidates), 1)
        self.assertTrue(state.company_candidates[0]["identity_verified"])
        self.assertEqual(state.company_candidates[0]["impact_ids"], ["IMP-001"])
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(all(item["status"] == "verified" for item in state.company_seed_audit))

    def test_existing_duplicate_seed_merges_without_reenrichment_and_conflict_is_excluded(self) -> None:
        for proposed_name, expected_status, expected_reason in (
            ("既有环保科技", "verified", "duplicate_existing_verified_identity"),
            ("冲突旧名称", "rejected", "duplicate_existing_name_code_conflict"),
        ):
            with self.subTest(proposed_name=proposed_name):
                profile_calls = 0

                def company_profile(**_kwargs):
                    nonlocal profile_calls
                    profile_calls += 1
                    return [{"main_business": "环保膜材料设备", "data_date": "2025"}]

                seed_payload = _seed_payload(
                    [
                        {
                            "impact_id": "IMP-001",
                            "proposed_name": proposed_name,
                            "historical_names": [],
                            "proposed_stock_code": "300777",
                            "seed_reason": "可能提供环保膜材料设备",
                            "origin_channels": ["llm"],
                        }
                    ]
                )
                client = RecordingLLMClient(
                    [
                        _react_finish(),
                        json.dumps(seed_payload, ensure_ascii=False),
                        json.dumps(
                            _company_payload_for("既有环保科技", "300777", "IMP-001"),
                            ensure_ascii=False,
                        ),
                    ]
                )
                invoker = _existing_candidate_invoker(company_profile)
                state = PolicyResearchState(
                    user_query="环保材料政策",
                    industry_impacts=[_existing_candidate_impact()],
                )

                output = run_llm_company_matcher(state, llm_client=client, mcp_invoker=invoker)

                self.assertEqual([company.company_name for company in output.companies], ["既有环保科技"])
                self.assertEqual(profile_calls, 1)
                self.assertEqual(len(state.company_candidates), 1)
                self.assertEqual(state.company_seed_audit[0]["status"], expected_status)
                self.assertEqual(state.company_seed_audit[0]["reason_code"], expected_reason)
                _system, seed_prompt = client.calls[1]
                self.assertIn('"remaining_deficit":3', seed_prompt)
                self.assertIn("既有环保科技", seed_prompt)
                self.assertIn("300777", seed_prompt)
                sources = {
                    str(item.get("source_type") or "")
                    for item in state.company_candidates[0].get("provenance") or []
                    if isinstance(item, dict)
                }
                if expected_status == "verified":
                    self.assertIn("llm_seed_existing_identity", sources)
                else:
                    self.assertNotIn("llm_seed_existing_identity", sources)

    def test_run_llm_company_matcher_rejects_malformed_json(self) -> None:
        store = build_sample_store()
        client = RecordingLLMClient([*[_react_finish()] * 5, "not json"])
        try:
            state = _state_with_industry_impacts(store)

            with self.assertRaises(StructuredOutputError):
                run_llm_company_matcher(state, llm_client=client, mcp_invoker=_fake_company_invoker())
        finally:
            store.close()

    def test_run_llm_company_matcher_rejects_company_outside_candidates(self) -> None:
        payload = _company_payload()
        payload["companies"][0]["company_name"] = "不存在的公司"
        store = build_sample_store()
        client = RecordingLLMClient([*[_react_finish()] * 5, json.dumps(payload, ensure_ascii=False)])
        try:
            state = _state_with_industry_impacts(store)

            with self.assertRaisesRegex(LLMCompanyMatchError, "outside candidate records"):
                run_llm_company_matcher(state, llm_client=client, mcp_invoker=_fake_company_invoker())
        finally:
            store.close()

    def test_run_llm_company_matcher_rejects_name_code_conflict(self) -> None:
        payload = _company_payload()
        payload["companies"][0]["stock_code"] = "300999"
        store = build_sample_store()
        client = RecordingLLMClient([*[_react_finish()] * 5, json.dumps(payload, ensure_ascii=False)])
        try:
            state = _state_with_industry_impacts(store)

            with self.assertRaisesRegex(LLMCompanyMatchError, "inconsistent name/code identity"):
                run_llm_company_matcher(state, llm_client=client, mcp_invoker=_fake_company_invoker())
        finally:
            store.close()

    def test_llm_and_deterministic_company_matcher_keep_default_runner_separate(self) -> None:
        store = build_sample_store()
        try:
            deterministic_state = _state_with_industry_impacts(store)
            llm_state = PolicyResearchState(
                user_query=deterministic_state.user_query,
                policy_ids=list(deterministic_state.policy_ids),
                policy_chunks=list(deterministic_state.policy_chunks),
                policy_analysis=dict(deterministic_state.policy_analysis),
                implementation_path=list(deterministic_state.implementation_path),
                industry_impacts=list(deterministic_state.industry_impacts),
                evidence=list(deterministic_state.evidence),
                uncertainties=list(deterministic_state.uncertainties),
            )
            run_company_matcher(deterministic_state, mcp_invoker=_fake_company_invoker())
            run_llm_company_matcher(
                llm_state,
                llm_client=_company_llm_client(),
                mcp_invoker=_fake_company_invoker(),
            )

            self.assertTrue(deterministic_state.company_matches)
            self.assertTrue(llm_state.company_matches)
            self.assertEqual(llm_state.company_matches[0]["company_name"], "清源模型安全科技")
        finally:
            store.close()

    def test_company_prompt_is_bounded_and_keeps_path_evidence_and_provenance(self) -> None:
        impacts: list[dict[str, object]] = []
        records: list[dict[str, object]] = []
        for impact_index in range(1, 6):
            impact_id = f"IMP-{impact_index:03d}"
            impacts.append(
                {
                    "impact_id": impact_id,
                    "industry": f"路径{impact_index}海水淡化设备",
                    "policy_measure": "建设海水淡化示范项目" * 30,
                    "implementation_action": "采购反渗透膜组件和高压泵" * 30,
                    "chain_segment": "反渗透膜组件与高压泵",
                    "transmission_logic": "项目投资传导至设备订单和膜组件需求" * 40,
                    "business_variables": ["膜组件需求", "高压泵订单"],
                    "affected_company_types": ["海水淡化设备供应商"],
                    "conditions": ["项目按期招标" * 20],
                    "risks": ["建设进度不确定" * 20],
                }
            )
            for company_index in range(1, 6):
                records.append(
                    {
                        "company_name": f"路径{impact_index}候选{company_index}",
                        "stock_code": f"{impact_index}{company_index:05d}"[-6:],
                        "impact_ids": [impact_id],
                        "industry_segment": "海水淡化设备",
                        "chain_segment": "反渗透膜组件",
                        "matched_business": "反渗透膜组件、高压泵和海水淡化成套设备" * 30,
                        "business_evidence": "公司公开资料显示主营海水淡化反渗透膜组件" * 40,
                        "negative_evidence": ["相关收入占比未披露" * 20],
                        "revenue_relevance": "unknown",
                        "data_date": "2025",
                        "provenance": [
                            {
                                "impact_id": impact_id,
                                "tool": "search_stock",
                                "tool_call_id": f"tool-{impact_index}-{company_index}",
                                "keyword": "反渗透膜组件",
                                "source_type": "cnfinancial_recall",
                            }
                        ],
                        "cnfinancial_raw": {"catalog": "绝不能进入 prompt" * 1000},
                    }
                )

        prompt = _render_company_prompt(impacts, records, [])
        rendered = prompt["system"] + prompt["user"]

        self.assertLessEqual(len(rendered), 16000)
        self.assertTrue(all(impact["impact_id"] in rendered for impact in impacts))
        self.assertTrue(all(record["company_name"] in rendered for record in records))
        self.assertIn("business_evidence", rendered)
        self.assertIn("negative_evidence", rendered)
        self.assertIn("provenance", rendered)
        self.assertIn("tool-1-1", rendered)
        self.assertNotIn("绝不能进入 prompt", rendered)

    def test_web_first_discovers_each_impact_without_legacy_recall_and_caches_exact_identity(self) -> None:
        impacts = [
            {**_web_first_impact(), "impact_id": "IMP-001", "industry": "路径一反渗透膜"},
            {**_web_first_impact(), "impact_id": "IMP-002", "industry": "路径二反渗透膜"},
        ]
        client = RecordingLLMClient(
            [
                json.dumps(_discovery_payload("IMP-001", ["路径一 反渗透膜 A股"]), ensure_ascii=False),
                json.dumps(_discovery_payload("IMP-002", ["路径二 反渗透膜 A股"]), ensure_ascii=False),
                json.dumps(_web_first_match_payload(["IMP-001", "IMP-002"]), ensure_ascii=False),
            ]
        )

        def web_search(*, arguments, **_kwargs):
            return [
                {
                    "company_name": "海膜科技",
                    "stock_code": "300123",
                    "title": "海膜科技反渗透膜业务",
                    "description": f"海膜科技 300123 主营反渗透膜；{arguments['query']}",
                    "url": f"https://exchange.example/{arguments['query']}",
                    "source_org": "交易所资料",
                    "date": "2026-08-01",
                }
            ]

        invoker = FakeMCPInvoker(
            {
                ("web-search", "search"): web_search,
                (CNFINANCIAL_SERVER, "search_stock"): [
                    {"company_name": "海膜科技", "stock_code": "300123"}
                ],
                (CNFINANCIAL_SERVER, "get_company_profile"): [
                    {"main_business": "反渗透膜", "data_date": "2025"}
                ],
                (CNFINANCIAL_SERVER, "get_company_info"): [
                    {
                        "company_name": "海膜科技",
                        "stock_code": "300123",
                        "listing_status": "active",
                        "data_date": "2026-08-01",
                    }
                ],
            }
        )
        state = PolicyResearchState(user_query="海水淡化政策", industry_impacts=impacts)

        with patch.dict(os.environ, {"POLICYCHAIN_COMPANY_DISCOVERY_MODE": "web_first"}):
            output = run_llm_company_matcher(state, llm_client=client, mcp_invoker=invoker)

        self.assertEqual(len(client.calls), 3)
        self.assertEqual({item.impact_id for item in output.companies}, {"IMP-001", "IMP-002"})
        self.assertEqual(len(state.company_candidates), 1)
        self.assertTrue(state.company_seeds)
        self.assertTrue(state.company_discovery_audit)
        self.assertTrue(state.company_identity_audit)
        self.assertTrue(state.company_evidence_bundles)
        tools = [call["tool_name"] for call in invoker.calls]
        self.assertNotIn("get_industry_list", tools)
        self.assertNotIn("get_concept_list", tools)
        self.assertNotIn("get_industry_stocks", tools)
        self.assertEqual(tools.count("search_stock"), 1)
        self.assertEqual(tools.count("get_company_profile"), 1)
        self.assertEqual(tools.count("get_company_info"), 1)
        self.assertEqual(
            [call["tool_name"] for call in invoker.calls if call["server_name"] == CNFINANCIAL_SERVER],
            ["search_stock", "get_company_info", "get_company_profile"],
        )
        self.assertTrue(
            all(call["arguments"] == {"keyword": "海膜科技"} for call in invoker.calls if call["tool_name"] == "search_stock")
        )
        self.assertTrue(all(item["coverage_status"] == "selected" for item in state.company_coverage))

    def test_web_first_exact_identity_tries_current_name_once_then_one_historical_alias(self) -> None:
        payload = _discovery_payload("IMP-001", ["反渗透膜 A股 公司公告"])
        payload["seeds"][0].update(
            {
                "proposed_name": "新海膜",
                "historical_names": ["海膜科技", "不得继续尝试的旧名称"],
            }
        )
        client = RecordingLLMClient(
            [
                json.dumps(payload, ensure_ascii=False),
                json.dumps(_web_first_match_payload(["IMP-001"], company_name="新海膜"), ensure_ascii=False),
            ]
        )

        def search_stock(*, arguments, **_kwargs):
            if arguments == {"keyword": "新海膜"}:
                return []
            if arguments == {"keyword": "海膜科技"}:
                return [{"company_name": "新海膜", "stock_code": "300123"}]
            self.fail(f"unexpected widened search: {arguments}")

        invoker = FakeMCPInvoker(
            {
                ("web-search", "search"): [],
                (CNFINANCIAL_SERVER, "search_stock"): search_stock,
                (CNFINANCIAL_SERVER, "get_company_profile"): [
                    {"main_business": "反渗透膜", "data_date": "2025"}
                ],
                (CNFINANCIAL_SERVER, "get_company_info"): [
                    {
                        "company_name": "新海膜",
                        "stock_code": "300123",
                        "listing_status": "active",
                        "data_date": "2026-08-01",
                    }
                ],
            }
        )
        state = PolicyResearchState(user_query="海水淡化政策", industry_impacts=[_web_first_impact()])

        with patch.dict(os.environ, {"POLICYCHAIN_COMPANY_DISCOVERY_MODE": "web_first"}):
            output = run_llm_company_matcher(state, llm_client=client, mcp_invoker=invoker)

        search_calls = [call["arguments"] for call in invoker.calls if call["tool_name"] == "search_stock"]
        self.assertEqual(search_calls, [{"keyword": "新海膜"}, {"keyword": "海膜科技"}])
        self.assertEqual(len(output.companies), 1)
        self.assertEqual(output.companies[0].stock_code, "300123")

    def test_web_first_allows_only_two_independent_web_sources_after_cnfinancial_technical_failure(self) -> None:
        client = RecordingLLMClient(
            [
                json.dumps(
                    _discovery_payload(
                        "IMP-001",
                        ["海膜科技 300123 反渗透膜 公告", "海膜科技 300123 反渗透膜 官网"],
                    ),
                    ensure_ascii=False,
                ),
                json.dumps(_web_first_match_payload(["IMP-001"], confidence=0.99), ensure_ascii=False),
            ]
        )

        def web_search(*, arguments, **_kwargs):
            if "公告" in arguments["query"]:
                return [
                    {
                        "company_name": "海膜科技",
                        "stock_code": "300123",
                        "title": "证券代码与主营业务公告",
                        "description": "海膜科技 300123 为上市公司，主营反渗透膜。",
                        "url": "https://one.example/notice/300123",
                        "source_org": "第一交易所",
                        "date": "2026-07-01",
                    }
                ]
            return [
                {
                    "company_name": "海膜科技",
                    "stock_code": "300123",
                    "title": "公司业务介绍",
                    "description": "海膜科技 股票代码300123，核心产品为反渗透膜。",
                    "url": "https://two.example/company/profile",
                    "source_org": "海膜科技官网",
                    "date": "2026-07-02",
                }
            ]

        def technical_failure(**_kwargs):
            raise MCPToolError("simulated timeout")

        invoker = FakeMCPInvoker(
            {
                ("web-search", "search"): web_search,
                (CNFINANCIAL_SERVER, "search_stock"): technical_failure,
                (CNFINANCIAL_SERVER, "get_company_profile"): technical_failure,
                (CNFINANCIAL_SERVER, "get_company_info"): technical_failure,
            }
        )
        state = PolicyResearchState(user_query="海水淡化", industry_impacts=[_web_first_impact()])
        recorder = RunRecorder(mode="llm")

        with patch.dict(os.environ, {"POLICYCHAIN_COMPANY_DISCOVERY_MODE": "web_first"}), recorder.activate():
            output = run_llm_company_matcher(state, llm_client=client, mcp_invoker=invoker)

        self.assertEqual(len(output.companies), 1)
        self.assertEqual(output.companies[0].match_level, "low")
        self.assertLessEqual(output.companies[0].confidence, 0.55)
        self.assertEqual(state.company_coverage[0]["coverage_status"], "web_fallback")
        self.assertIn("CNFinancial 未完成交叉验证", " ".join(output.uncertainties))
        events = [json.loads(line) for line in recorder.events_path.read_text(encoding="utf-8").splitlines()]
        event_types = [event["event_type"] for event in events]
        for required in (
            "company.discovery",
            "company.seed",
            "company.identity",
            "company.enrichment",
            "company.audit",
            "company.rank",
        ):
            self.assertIn(required, event_types)
        for event in events:
            if event["event_type"] in {
                "company.discovery",
                "company.seed",
                "company.identity",
                "company.enrichment",
                "company.audit",
                "company.rank",
            }:
                self.assertEqual(event["run_id"], recorder.run_id)
                for field in ("impact_id", "seed_id", "tool_call_id", "source", "reason_code", "cache_hit"):
                    self.assertIn(field, event)

    def test_web_first_single_or_duplicate_web_source_stays_unresolved(self) -> None:
        client = RecordingLLMClient(
            json.dumps(
                _discovery_payload("IMP-001", ["海膜科技 300123 公告", "海膜科技 300123 官网"]),
                ensure_ascii=False,
            )
        )

        def duplicated_web(**_kwargs):
            return [
                {
                    "company_name": "海膜科技",
                    "stock_code": "300123",
                    "title": "同源转载",
                    "description": "海膜科技 300123 主营反渗透膜。",
                    "url": "https://same.example/article",
                    "source_org": "同一来源",
                }
            ]

        def technical_failure(**_kwargs):
            raise MCPToolError("circuit open")

        invoker = FakeMCPInvoker(
            {
                ("web-search", "search"): duplicated_web,
                (CNFINANCIAL_SERVER, "search_stock"): technical_failure,
                (CNFINANCIAL_SERVER, "get_company_profile"): technical_failure,
                (CNFINANCIAL_SERVER, "get_company_info"): technical_failure,
            }
        )
        state = PolicyResearchState(user_query="海水淡化", industry_impacts=[_web_first_impact()])

        with patch.dict(os.environ, {"POLICYCHAIN_COMPANY_DISCOVERY_MODE": "web_first"}):
            output = run_llm_company_matcher(state, llm_client=client, mcp_invoker=invoker)

        self.assertEqual(output.companies, [])
        self.assertEqual(state.company_candidates, [])
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(
            any(item.get("reason_code") == "web_fallback_insufficient_independent_sources" for item in state.company_identity_audit)
        )

    def test_web_first_successful_cnfinancial_empty_does_not_use_web_fallback(self) -> None:
        client = RecordingLLMClient(
            json.dumps(
                _discovery_payload("IMP-001", ["海膜科技 300123 公告", "海膜科技 300123 官网"]),
                ensure_ascii=False,
            )
        )

        def web_search(*, arguments, **_kwargs):
            domain = "one.example" if "公告" in arguments["query"] else "two.example"
            return [
                {
                    "company_name": "海膜科技",
                    "stock_code": "300123",
                    "description": "海膜科技 300123 主营反渗透膜。",
                    "url": f"https://{domain}/evidence",
                    "source_org": domain,
                }
            ]

        def technical_failure(**_kwargs):
            raise MCPToolError("simulated timeout")

        invoker = FakeMCPInvoker(
            {
                ("web-search", "search"): web_search,
                (CNFINANCIAL_SERVER, "search_stock"): [],
                (CNFINANCIAL_SERVER, "get_company_profile"): technical_failure,
                (CNFINANCIAL_SERVER, "get_company_info"): technical_failure,
            }
        )
        state = PolicyResearchState(user_query="海水淡化", industry_impacts=[_web_first_impact()])

        with patch.dict(os.environ, {"POLICYCHAIN_COMPANY_DISCOVERY_MODE": "web_first"}):
            output = run_llm_company_matcher(state, llm_client=client, mcp_invoker=invoker)

        self.assertEqual(output.companies, [])
        self.assertEqual(state.company_coverage[0]["coverage_status"], "cnfinancial_empty")
        self.assertEqual(len(client.calls), 1)
        self.assertFalse(any(item.get("reason_code") == "web_fallback" for item in state.company_identity_audit))

    def test_web_first_identity_conflict_is_permanently_rejected(self) -> None:
        client = RecordingLLMClient(json.dumps(_discovery_payload("IMP-001", []), ensure_ascii=False))
        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "search_stock"): [
                    {"company_name": "海膜科技", "stock_code": "300999"}
                ],
            }
        )
        state = PolicyResearchState(user_query="海水淡化", industry_impacts=[_web_first_impact()])

        with patch.dict(os.environ, {"POLICYCHAIN_COMPANY_DISCOVERY_MODE": "web_first"}):
            output = run_llm_company_matcher(state, llm_client=client, mcp_invoker=invoker)

        self.assertEqual(output.companies, [])
        self.assertTrue(any(item.get("reason_code") == "identity_conflict" for item in state.company_identity_audit))
        self.assertNotIn("get_company_profile", [call["tool_name"] for call in invoker.calls])
        self.assertEqual(state.company_coverage[0]["coverage_status"], "identity_conflict")
        self.assertEqual(len(client.calls), 1)

    def test_web_first_schema_failure_does_not_silently_call_legacy_recall(self) -> None:
        client = RecordingLLMClient("not json")
        invoker = FakeMCPInvoker({})
        state = PolicyResearchState(user_query="海水淡化", industry_impacts=[_web_first_impact()])

        with patch.dict(os.environ, {"POLICYCHAIN_COMPANY_DISCOVERY_MODE": "web_first"}):
            output = run_llm_company_matcher(state, llm_client=client, mcp_invoker=invoker)

        self.assertEqual(output.companies, [])
        self.assertEqual(invoker.calls, [])
        self.assertEqual(state.company_coverage[0]["coverage_status"], "discovery_error")
        self.assertIn("未回退到旧 CNFinancial-first", " ".join(state.uncertainties))

    def test_web_first_evaluation_runs_after_enrichment_and_prompt_contains_only_verified_bundles(self) -> None:
        timeline: list[str] = []
        discovery = _discovery_payload("IMP-001", ["反渗透膜 A股 公司业务"])
        discovery["seeds"].append(
            {
                "impact_id": "IMP-001",
                "proposed_name": "拒绝泄漏公司",
                "historical_names": [],
                "proposed_stock_code": "999999",
                "seed_reason": "不得进入评价 prompt",
                "origin_channels": ["llm"],
            }
        )

        class OrderedClient(RecordingLLMClient):
            def generate(self, system_prompt: str, user_prompt: str) -> str:
                timeline.append("llm:company_matcher" if "Company Matcher" in system_prompt else "llm:discovery")
                return super().generate(system_prompt, user_prompt)

        client = OrderedClient(
            [
                json.dumps(discovery, ensure_ascii=False),
                json.dumps(_web_first_match_payload(["IMP-001"], confidence=0.99), ensure_ascii=False),
            ]
        )

        def web_search(**_kwargs):
            timeline.append("web:search")
            return [
                {
                    "company_name": "海膜科技",
                    "stock_code": "300123",
                    "description": "海膜科技 300123 主营反渗透膜。",
                    "url": "https://verified.example/company/300123",
                    "source_org": "验证来源",
                    "date": "2026-08-01",
                },
                {
                    "company_name": "拒绝泄漏公司",
                    "stock_code": "999999",
                    "description": "拒绝泄漏公司 999999 的无效资料。",
                    "url": "https://rejected.example/company/999999",
                    "source_org": "拒绝来源",
                    "date": "2026-08-01",
                },
            ]

        def search_stock(**_kwargs):
            timeline.append("cnfinancial:search_stock")
            return [{"company_name": "海膜科技", "stock_code": "300123"}]

        def company_info(**_kwargs):
            timeline.append("cnfinancial:get_company_info")
            return [
                {
                    "company_name": "海膜科技",
                    "stock_code": "300123",
                    "listing_status": "active",
                    "data_date": "2026-08-01",
                }
            ]

        def company_profile(**_kwargs):
            timeline.append("cnfinancial:get_company_profile")
            return [{"main_business": "反渗透膜", "data_date": "2025"}]

        invoker = FakeMCPInvoker(
            {
                ("web-search", "search"): web_search,
                (CNFINANCIAL_SERVER, "search_stock"): search_stock,
                (CNFINANCIAL_SERVER, "get_company_info"): company_info,
                (CNFINANCIAL_SERVER, "get_company_profile"): company_profile,
            }
        )
        state = PolicyResearchState(user_query="海水淡化", industry_impacts=[_web_first_impact()])

        with patch.dict(os.environ, {"POLICYCHAIN_COMPANY_DISCOVERY_MODE": "web_first"}):
            output = run_llm_company_matcher(state, llm_client=client, mcp_invoker=invoker)

        self.assertEqual(
            timeline,
            [
                "llm:discovery",
                "web:search",
                "cnfinancial:search_stock",
                "cnfinancial:get_company_info",
                "cnfinancial:get_company_profile",
                "llm:company_matcher",
            ],
        )
        evaluation_prompt = client.calls[-1][1]
        for expected in (
            "海膜科技",
            "300123",
            "IMP-001",
            "path_specific_business",
            "tool_status",
            "tool_call_id",
            "cnfinancial_info_evidence",
            "cnfinancial_profile_evidence",
            "web_evidence",
        ):
            self.assertIn(expected, evaluation_prompt)
        self.assertNotIn("拒绝泄漏公司", evaluation_prompt)
        self.assertNotIn("999999", evaluation_prompt)
        self.assertEqual(len(output.companies), 1)
        self.assertLessEqual(output.companies[0].confidence, 0.92)

    def test_web_first_evaluation_rejects_invented_company_and_path_before_deterministic_audit(self) -> None:
        evaluation = _web_first_match_payload(["IMP-001"], confidence=0.99)
        invented_company = dict(evaluation["companies"][0])
        invented_company.update({"company_name": "虚构公司", "stock_code": "300999"})
        invented_path = dict(evaluation["companies"][0])
        invented_path["impact_id"] = "IMP-999"
        evaluation["companies"].extend([invented_company, invented_path])
        client = RecordingLLMClient(
            [
                json.dumps(_discovery_payload("IMP-001", []), ensure_ascii=False),
                json.dumps(evaluation, ensure_ascii=False),
            ]
        )
        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "search_stock"): [
                    {"company_name": "海膜科技", "stock_code": "300123"}
                ],
                (CNFINANCIAL_SERVER, "get_company_info"): [
                    {
                        "company_name": "海膜科技",
                        "stock_code": "300123",
                        "listing_status": "active",
                        "data_date": "2026-08-01",
                    }
                ],
                (CNFINANCIAL_SERVER, "get_company_profile"): [
                    {"main_business": "反渗透膜", "data_date": "2025"}
                ],
            }
        )
        state = PolicyResearchState(user_query="海水淡化", industry_impacts=[_web_first_impact()])

        with patch.dict(os.environ, {"POLICYCHAIN_COMPANY_DISCOVERY_MODE": "web_first"}):
            output = run_llm_company_matcher(state, llm_client=client, mcp_invoker=invoker)

        self.assertEqual([(item.company_name, item.impact_id) for item in output.companies], [("海膜科技", "IMP-001")])
        self.assertLessEqual(output.companies[0].confidence, 0.92)
        reason_codes = {str(item.get("reason_code") or "") for item in state.company_match_audit}
        self.assertIn("llm_company_not_whitelisted", reason_codes)
        self.assertIn("llm_path_not_whitelisted", reason_codes)
        self.assertNotIn("虚构公司", json.dumps(state.company_matches, ensure_ascii=False))


def _state_with_industry_impacts(store) -> PolicyResearchState:
    state = PolicyResearchState(user_query="生成式人工智能服务提供者有哪些管理要求")
    run_policy_analyst(state, store)
    run_impact_analyst(state)
    return state


def _react_finish() -> str:
    return json.dumps({"thought": "enough evidence", "action": "finish", "arguments": {}}, ensure_ascii=False)


def _company_llm_client() -> RecordingLLMClient:
    return RecordingLLMClient(
        [
            *[_react_finish()] * 5,
            *[json.dumps(_seed_payload(), ensure_ascii=False)] * 5,
            json.dumps(_company_payload(), ensure_ascii=False),
        ]
    )


def _fake_company_invoker() -> FakeMCPInvoker:
    return FakeMCPInvoker(
        {
            (CNFINANCIAL_SERVER, "get_industry_list"): [{"名称": "软件开发"}],
            (CNFINANCIAL_SERVER, "get_concept_list"): [{"名称": "人工智能"}],
            (CNFINANCIAL_SERVER, "get_industry_stocks"): [
                {
                    "company_name": "清源模型安全科技",
                    "stock_code": "300001",
                    "main_business": "模型安全评估平台",
                    "description": "提供模型安全评估和训练数据质量检测服务。",
                }
            ],
            (CNFINANCIAL_SERVER, "get_company_profile"): [
                {
                    "main_business": "模型安全评估平台",
                    "revenue_ratio": "28%",
                }
            ],
        }
    )


def _existing_candidate_invoker(company_profile) -> FakeMCPInvoker:
    return FakeMCPInvoker(
        {
            (CNFINANCIAL_SERVER, "get_industry_list"): [{"名称": "环保材料"}],
            (CNFINANCIAL_SERVER, "get_concept_list"): [],
            (CNFINANCIAL_SERVER, "get_industry_stocks"): [
                {
                    "company_name": "既有环保科技",
                    "stock_code": "300777",
                    "main_business": "环保膜材料设备",
                }
            ],
            (CNFINANCIAL_SERVER, "search_stock"): [],
            (CNFINANCIAL_SERVER, "get_company_profile"): company_profile,
            ("web-search", "search"): [
                {
                    "title": "既有环保科技主营业务",
                    "description": "公司主营环保膜材料设备。",
                    "url": "https://www.cninfo.com.cn/fake/300777",
                    "date": "2026-07-01",
                }
            ],
        }
    )


def _existing_candidate_impact() -> dict[str, object]:
    return {
        "impact_id": "IMP-001",
        "industry": "环保材料",
        "chain_segment": "环保膜材料设备",
        "transmission_logic": "环保项目采购带动膜材料设备需求",
        "business_variables": ["环保材料设备需求"],
        "affected_company_types": ["环保材料设备供应商"],
        "conditions": [],
        "risks": [],
    }


def _company_payload() -> dict[str, object]:
    return {
        "companies": [
            {
                "company_name": "清源模型安全科技",
                "impact_id": "IMP-002",
                "industry_segment": "算法模型研发与评估",
                "matched_business": "提供大模型安全评估、训练数据质量检测和模型风险测试服务。",
                "match_level": "high",
                "business_evidence": [
                    {
                        "source_name": "Mock Company Profile",
                        "source_url": "mock://company/qingyuan-model-safety",
                        "text": "公司资料显示其核心服务包括模型安全评估、训练数据质量检测和生成式人工智能风险测试。",
                        "data_date": "2026-01-15",
                    }
                ],
                "policy_link": "政策要求模型、算法和训练数据环节承担安全治理责任。",
                "revenue_relevance": "medium",
                "conditions": ["需核验真实官网和公告资料。"],
                "risks": ["本地 mock 数据不可代表真实公司资料。"],
                "data_date": "2026-01-15",
                "confidence": 0.86,
            }
        ],
        "uncertainties": ["公司资料来自本地 mock 数据，仅用于验证流程。"],
    }


def _seed_payload(seeds: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {"seeds": list(seeds or []), "uncertainties": []}


def _discovery_payload(impact_id: str, web_queries: list[str]) -> dict[str, object]:
    return {
        "impact_id": impact_id,
        "web_queries": web_queries,
        "seeds": [
            {
                "impact_id": impact_id,
                "proposed_name": "海膜科技",
                "historical_names": [],
                "proposed_stock_code": "300123",
                "seed_reason": "公司可能提供反渗透膜。",
                "origin_channels": ["llm"],
            }
        ],
        "uncertainties": [],
    }


def _web_first_match_payload(
    impact_ids: list[str],
    *,
    company_name: str = "海膜科技",
    stock_code: str = "300123",
    confidence: float = 0.9,
) -> dict[str, object]:
    return {
        "companies": [
            {
                "company_name": company_name,
                "stock_code": stock_code,
                "industry_segment": "海水淡化反渗透膜",
                "impact_id": impact_id,
                "impact_industry": "海水淡化反渗透膜",
                "chain_segment": "反渗透膜",
                "matched_business": "公司主营反渗透膜。",
                "related_product_or_business": "反渗透膜",
                "match_level": "high",
                "revenue_or_ratio": "",
                "source_url": "https://verified.example/company/300123",
                "match_conditions": ["需持续核对公开业务资料。"],
                "negative_evidence": [],
                "business_evidence": [
                    {
                        "source_name": "验证后 evidence bundle",
                        "source_url": "https://verified.example/company/300123",
                        "text": "公司主营反渗透膜。",
                        "data_date": "2026-08-01",
                    }
                ],
                "policy_link": "示范项目采购带动反渗透膜需求。",
                "revenue_relevance": "unknown",
                "conditions": ["政策项目实际落地。"],
                "risks": ["需求传导存在不确定性。"],
                "data_date": "2026-08-01",
                "confidence": confidence,
                "audit_status": "pending",
                "audit_reason": "等待系统确定性后审计。",
            }
            for impact_id in impact_ids
        ],
        "uncertainties": [],
    }


def _seed_test_impact() -> dict[str, object]:
    return {
        "impact_id": "IMP-001",
        "industry": "环保材料设备",
        "chain_segment": "环保膜材料",
        "transmission_logic": "项目采购带动环保膜材料需求",
        "business_variables": ["环保材料需求"],
        "affected_company_types": ["环保材料供应商"],
        "conditions": [],
        "risks": [],
    }


def _web_first_impact() -> dict[str, object]:
    return {
        "impact_id": "IMP-001",
        "industry": "海水淡化反渗透膜",
        "chain_segment": "反渗透膜",
        "transmission_logic": "示范项目采购带动反渗透膜需求",
        "business_variables": ["反渗透膜"],
        "affected_company_types": ["反渗透膜供应商"],
        "conditions": [],
        "risks": [],
    }


def _verified_record(index: int) -> dict[str, object]:
    return {
        "company_name": f"已验证企业{index}",
        "stock_code": f"300{index:03d}",
        "impact_ids": ["IMP-001"],
        "identity_verified": True,
        "candidate_source_tool": "get_industry_stocks",
        "matched_business": "环保膜材料设备",
        "business_evidence": "主营环保膜材料设备",
        "provenance": [{"impact_id": "IMP-001", "tool": "get_industry_stocks"}],
    }


def _company_payload_for(name: str, code: str, impact_id: str) -> dict[str, object]:
    payload = _company_payload()
    company = payload["companies"][0]
    assert isinstance(company, dict)
    company.update(
        {
            "company_name": name,
            "stock_code": code,
            "impact_id": impact_id,
            "industry_segment": "海水淡化设备",
            "chain_segment": "环保反渗透膜材料",
            "matched_business": "环保反渗透膜材料与海水淡化设备",
            "related_product_or_business": "环保反渗透膜材料",
            "policy_link": "示范项目采购带动环保反渗透膜材料需求",
            "business_evidence": [
                {
                    "source_name": "Mock Company Profile",
                    "source_url": "https://www.cninfo.com.cn/fake/300123",
                    "text": "公司主营环保反渗透膜材料与海水淡化设备。",
                    "data_date": "2025",
                }
            ],
            "confidence": 0.86,
        }
    )
    return payload


if __name__ == "__main__":
    unittest.main()
