from __future__ import annotations

import unittest

from policychain.prompts import PROMPT_TEMPLATES, get_prompt_template, render_prompt


class PromptTests(unittest.TestCase):
    def test_all_core_agent_templates_exist(self) -> None:
        self.assertEqual(
            set(PROMPT_TEMPLATES),
            {"policy_analyst", "impact_analyst", "company_matcher", "report_writer"},
        )

    def test_policy_analyst_prompt_renders_inputs_and_boundaries(self) -> None:
        rendered = render_prompt(
            "policy_analyst",
            user_query="生成式人工智能服务要求",
            metadata={"policy_id": "POL-2023-NAT-0048"},
            chunks=[{"chunk_id": "POL-2023-NAT-0048-S001-C001", "content": "应当备案"}],
        )

        self.assertIn("生成式人工智能服务要求", rendered["user"])
        self.assertIn("POL-2023-NAT-0048", rendered["user"])
        self.assertIn("证据", rendered["system"])
        self.assertIn("不确定性", rendered["system"])
        self.assertIn("禁止输出买入", rendered["system"])
        self.assertIn('"policy_goals"', rendered["user"])
        self.assertIn('"target_entities"', rendered["user"])
        self.assertIn('"strength_assessment"', rendered["user"])
        self.assertIn("不得使用 policy_objectives", rendered["user"])
        self.assertEqual(rendered["output_schema_name"], "PolicyAnalysisOutput")

    def test_company_matcher_prompt_keeps_business_matching_boundary(self) -> None:
        rendered = render_prompt(
            "company_matcher",
            industry_impacts=[{"industry": "生成式人工智能服务"}],
            company_records=[{"company_name": "示例公司"}],
        )

        self.assertIn("业务相关性匹配", rendered["system"])
        self.assertIn("公司业务匹配", rendered["user"])
        self.assertIn("不得把业务相关性写成投资结论", rendered["user"])
        self.assertIn('"companies"', rendered["user"])
        self.assertIn('"company_name"', rendered["user"])
        self.assertIn("不得编造公司", rendered["user"])
        self.assertIn("逐条绑定到行业路径", rendered["user"])
        self.assertIn("合理性审查", rendered["system"])
        self.assertIn('"impact_id"', rendered["user"])

    def test_report_writer_prompt_requires_detail_and_full_path_coverage(self) -> None:
        rendered = render_prompt(
            "report_writer",
            policy_analysis={"policy_identity": {"title": "测试政策"}},
            impact_analysis={"industry_impacts": [{"industry": "钢铁"}, {"industry": "算力"}]},
            company_matches={"company_coverage": []},
            evidence={},
            uncertainties=[],
        )

        self.assertIn("较完整的政策研究说明", rendered["system"])
        self.assertIn("必须覆盖所有行业影响路径", rendered["system"])
        self.assertIn("没有可靠公司匹配的路径也要说明原因", rendered["system"])

    def test_impact_analyst_prompt_includes_exact_json_contract(self) -> None:
        rendered = render_prompt(
            "impact_analyst",
            policy_analysis={"policy_identity": {"policy_id": "POL-2023-NAT-0048"}},
            policy_chunks=[{"chunk_id": "POL-2023-NAT-0048-S001-C001"}],
        )

        self.assertIn('"implementation_chain"', rendered["user"])
        self.assertIn('"industry_impacts"', rendered["user"])
        self.assertIn('"impact_type"', rendered["user"])
        self.assertIn("不得输出额外字段", rendered["user"])

    def test_unknown_prompt_name_fails_clearly(self) -> None:
        with self.assertRaisesRegex(KeyError, "Unknown prompt template"):
            get_prompt_template("unknown")

    def test_missing_prompt_variable_fails_clearly(self) -> None:
        with self.assertRaisesRegex(KeyError, "Missing prompt variable"):
            render_prompt("impact_analyst", policy_analysis={})


if __name__ == "__main__":
    unittest.main()
