from __future__ import annotations

import json
import unittest

from policychain.safety import SafetyViolation
from policychain.schemas.agent_outputs import (
    CompanyDiscoveryOutput,
    CompanyMatchOutput,
    CompanySeedOutput,
    ImpactAnalysisOutput,
    PolicyAnalysisOutput,
)
from policychain.structured_output import (
    StructuredOutputError,
    parse_json_object,
    parse_structured_output,
    validate_structured_payload,
)


class StructuredOutputTests(unittest.TestCase):
    def test_parse_json_object_accepts_raw_json(self) -> None:
        self.assertEqual(parse_json_object('{"a": 1}'), {"a": 1})

    def test_parse_json_object_accepts_fenced_json(self) -> None:
        self.assertEqual(parse_json_object('```json\n{"a": 1}\n```'), {"a": 1})

    def test_parse_json_object_extracts_embedded_json_object(self) -> None:
        self.assertEqual(parse_json_object('前缀 {"a": {"b": 2}} 后缀'), {"a": {"b": 2}})

    def test_parse_json_object_rejects_malformed_json(self) -> None:
        with self.assertRaisesRegex(StructuredOutputError, "valid JSON"):
            parse_json_object("不是 JSON")

    def test_parse_structured_policy_analysis_output(self) -> None:
        output = parse_structured_output(json.dumps(_policy_payload(), ensure_ascii=False), "PolicyAnalysisOutput")

        self.assertIsInstance(output, PolicyAnalysisOutput)
        data = output.to_dict()
        self.assertEqual(data["policy_identity"]["policy_id"], "POL-2023-NAT-0048")
        self.assertEqual(data["strength_assessment"]["level"], "medium")
        self.assertEqual(data["evidence"][0]["chunk_id"], "POL-2023-NAT-0048-S001-C001")

    def test_policy_historical_changes_accepts_object_items_from_llm(self) -> None:
        payload = _policy_payload()
        payload["historical_changes"] = [
            {
                "policy_title": "生成式人工智能服务管理暂行办法",
                "date": "2023-07-10",
                "change": "明确服务提供者安全义务",
                "evidence": ["第一条", "第九条"],
            }
        ]

        output = validate_structured_payload(payload, "PolicyAnalysisOutput")

        self.assertEqual(len(output.historical_changes), 1)
        self.assertIn("生成式人工智能服务管理暂行办法", output.historical_changes[0])
        self.assertIn("明确服务提供者安全义务", output.historical_changes[0])

    def test_parse_structured_impact_analysis_output(self) -> None:
        output = parse_structured_output(json.dumps(_impact_payload(), ensure_ascii=False), "ImpactAnalysisOutput")

        self.assertIsInstance(output, ImpactAnalysisOutput)
        data = output.to_dict()
        self.assertEqual(data["implementation_chain"][0]["step_index"], 1)
        self.assertEqual(data["industry_impacts"][0]["impact_type"], "direct")

    def test_parse_structured_company_match_output(self) -> None:
        output = parse_structured_output(json.dumps(_company_payload(), ensure_ascii=False), "CompanyMatchOutput")

        self.assertIsInstance(output, CompanyMatchOutput)
        data = output.to_dict()
        self.assertEqual(data["companies"][0]["company_name"], "示例公司")
        self.assertEqual(data["companies"][0]["confidence"], 0.82)

    def test_parse_structured_company_seed_output(self) -> None:
        output = parse_structured_output(json.dumps(_company_seed_payload(), ensure_ascii=False), "CompanySeedOutput")

        self.assertIsInstance(output, CompanySeedOutput)
        self.assertEqual(output.seeds[0].impact_id, "IMP-001")
        self.assertEqual(output.seeds[0].proposed_stock_code, "300123")
        self.assertEqual(output.seeds[0].historical_names, ["示例旧名"])

    def test_company_discovery_output_enforces_path_queries_and_seed_budget(self) -> None:
        payload = {
            "impact_id": "IMP-001",
            "web_queries": ["反渗透膜 A股 公司 主营业务", "海水淡化设备 证券代码 公告"],
            "seeds": _company_seed_payload()["seeds"],
            "uncertainties": [],
        }

        output = validate_structured_payload(payload, "CompanyDiscoveryOutput")

        self.assertIsInstance(output, CompanyDiscoveryOutput)
        self.assertEqual(output.impact_id, "IMP-001")
        self.assertEqual(len(output.web_queries), 2)

        payload["web_queries"].append("第三条查询")
        with self.assertRaisesRegex(StructuredOutputError, "at most 2"):
            validate_structured_payload(payload, "CompanyDiscoveryOutput")

        mismatch = {
            **payload,
            "web_queries": [],
            "seeds": [{**_company_seed_payload()["seeds"][0], "impact_id": "IMP-002"}],
        }
        with self.assertRaisesRegex(StructuredOutputError, "match the top-level"):
            validate_structured_payload(mismatch, "CompanyDiscoveryOutput")

    def test_company_seed_output_enforces_six_per_impact_and_three_historical_names(self) -> None:
        too_many = _company_seed_payload()
        too_many["seeds"] = [dict(too_many["seeds"][0]) for _ in range(7)]
        with self.assertRaisesRegex(StructuredOutputError, "at most 6"):
            validate_structured_payload(too_many, "CompanySeedOutput")

        aliases = _company_seed_payload()
        aliases["seeds"][0]["historical_names"] = ["旧名一", "旧名二", "旧名三", "旧名四"]
        with self.assertRaisesRegex(StructuredOutputError, "historical_names"):
            validate_structured_payload(aliases, "CompanySeedOutput")

    def test_company_seed_output_rejects_invalid_code_or_extra_seed_fields(self) -> None:
        invalid_code = _company_seed_payload()
        invalid_code["seeds"][0]["proposed_stock_code"] = "ABC123"
        with self.assertRaisesRegex(StructuredOutputError, "six-digit"):
            validate_structured_payload(invalid_code, "CompanySeedOutput")

        extra = _company_seed_payload()
        extra["seeds"][0]["verified"] = True
        with self.assertRaisesRegex(StructuredOutputError, "unsupported field"):
            validate_structured_payload(extra, "CompanySeedOutput")

    def test_company_uncertainties_accept_object_items_from_llm(self) -> None:
        payload = _company_payload()
        payload["uncertainties"] = [
            {
                "reason": "CNFinancial 未返回分部收入",
                "evidence": ["仅有主营业务描述", "缺少收入占比"],
            }
        ]

        output = validate_structured_payload(payload, "CompanyMatchOutput")

        self.assertEqual(len(output.uncertainties), 1)
        self.assertIn("CNFinancial 未返回分部收入", output.uncertainties[0])
        self.assertIn("缺少收入占比", output.uncertainties[0])

    def test_company_evidence_empty_data_date_becomes_unknown(self) -> None:
        payload = _company_payload()
        payload["companies"][0]["business_evidence"][0]["data_date"] = ""

        output = validate_structured_payload(payload, "CompanyMatchOutput")

        self.assertEqual(output.companies[0].business_evidence[0].data_date, "unknown")

    def test_validate_structured_payload_rejects_missing_required_fields(self) -> None:
        payload = _policy_payload()
        del payload["evidence"]

        with self.assertRaisesRegex(StructuredOutputError, "missing required field"):
            validate_structured_payload(payload, "PolicyAnalysisOutput")

    def test_validate_structured_payload_rejects_invalid_enum(self) -> None:
        payload = _impact_payload()
        payload["industry_impacts"][0]["impact_type"] = "certain"

        with self.assertRaisesRegex(StructuredOutputError, "impact_type"):
            validate_structured_payload(payload, "ImpactAnalysisOutput")

    def test_validate_structured_payload_rejects_invalid_confidence(self) -> None:
        payload = _company_payload()
        payload["companies"][0]["confidence"] = 1.2

        with self.assertRaisesRegex(StructuredOutputError, "confidence"):
            validate_structured_payload(payload, "CompanyMatchOutput")

    def test_parse_structured_output_rejects_prohibited_terms(self) -> None:
        payload = _company_payload()
        payload["companies"][0]["policy_link"] = "业务相关性不能写成推荐股票"

        with self.assertRaises(SafetyViolation):
            parse_structured_output(json.dumps(payload, ensure_ascii=False), "CompanyMatchOutput")

    def test_unsupported_schema_fails_clearly(self) -> None:
        with self.assertRaisesRegex(StructuredOutputError, "Unsupported structured output schema"):
            validate_structured_payload({}, "UnknownOutput")


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
            "reasons": ["文本包含明确义务要求"],
            "uncertainties": ["尚未纳入配套细则"],
        },
        "evidence": [_evidence()],
        "uncertainties": ["仅基于样例政策文本"],
    }


def _impact_payload() -> dict[str, object]:
    return {
        "implementation_actors": ["生成式人工智能服务提供者"],
        "implementation_mechanisms": ["备案与合规审查"],
        "implementation_chain": [
            {
                "step_index": 1,
                "actor": "生成式人工智能服务提供者",
                "action": "落实内容安全要求",
                "mechanism": "备案与合规审查",
                "evidence": [_evidence()],
            }
        ],
        "industry_impacts": [
            {
                "industry": "生成式人工智能服务",
                "impact_type": "direct",
                "direction": "mixed",
                "transmission_logic": "政策直接影响服务提供和安全治理能力建设",
                "conditions": ["需结合监管执行口径"],
                "risks": ["合规能力不足会提高整改压力"],
                "evidence": [_evidence()],
            }
        ],
        "uncertainties": ["尚未接入产业数据"],
        "evidence": [_evidence()],
    }


def _company_payload() -> dict[str, object]:
    return {
        "companies": [
            {
                "company_name": "示例公司",
                "industry_segment": "生成式人工智能服务",
                "matched_business": "大模型服务合规工具",
                "match_level": "high",
                "business_evidence": [
                    {
                        "source_name": "mock company source",
                        "source_url": "https://example.test/company",
                        "text": "示例公司提供大模型内容安全能力",
                        "data_date": "2026-01-01",
                    }
                ],
                "policy_link": "政策要求提升内容安全和模型治理能力",
                "revenue_relevance": "unknown",
                "conditions": ["需核验真实公告和官网"],
                "risks": ["本地 mock 数据不可代表真实公司资料"],
                "data_date": "2026-01-01",
                "confidence": 0.82,
            }
        ],
        "uncertainties": ["真实业务匹配需接入公开资料"],
    }


def _company_seed_payload() -> dict[str, object]:
    return {
        "seeds": [
            {
                "impact_id": "IMP-001",
                "proposed_name": "示例科技",
                "historical_names": ["示例旧名"],
                "proposed_stock_code": "300123",
                "seed_reason": "公开线索显示其具体设备业务可能对应路径",
                "origin_channels": ["llm", "web"],
            }
        ],
        "uncertainties": ["seed 尚未完成身份或业务验证"],
    }


def _evidence() -> dict[str, object]:
    return {
        "policy_id": "POL-2023-NAT-0048",
        "chunk_id": "POL-2023-NAT-0048-S001-C001",
        "source_url": "https://example.test/policy",
        "text": "服务提供者应当依法履行安全义务",
        "note": "第一条",
    }


if __name__ == "__main__":
    unittest.main()
