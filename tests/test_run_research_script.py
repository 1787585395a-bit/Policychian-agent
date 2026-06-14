from __future__ import annotations

from contextlib import redirect_stderr
import io
from pathlib import Path
import json
import unittest
from unittest.mock import patch

from policychain.paths import FULL_DB_PATH, SAMPLE_DB_PATH
from scripts.ingest_sample import ingest_sample_database
from scripts.run_research import _selected_db_path, parse_args, run_research
from tests.helpers import artifact_db_path


PROHIBITED_TERMS = ("买入", "卖出", "目标价", "推荐股票")


class RunResearchScriptTests(unittest.TestCase):
    def test_run_research_builds_db_and_returns_report(self) -> None:
        db_path = artifact_db_path("run_research")

        report = run_research(
            query="生成式人工智能服务提供者有哪些管理要求",
            db_path=db_path,
            ensure_sample_db=True,
            rebuild_sample_db=True,
        )

        self.assertIn("PolicyChain 政策研究报告", report)
        self.assertIn("生成式人工智能服务管理暂行办法", report)
        self.assertIn("公司业务匹配清单", report)

    def test_run_research_can_write_report_file(self) -> None:
        db_path = artifact_db_path("run_research_out")
        output_path = Path("artifacts/test-results/run_research_report.md")

        report = run_research(
            query="生成式人工智能 公司影响",
            db_path=db_path,
            ensure_sample_db=True,
            rebuild_sample_db=True,
            output_path=output_path,
        )

        self.assertTrue(output_path.exists())
        self.assertEqual(output_path.read_text(encoding="utf-8"), report)
        for term in PROHIBITED_TERMS:
            self.assertNotIn(term, report)

    def test_run_research_can_use_injected_llm_workflow(self) -> None:
        db_path = artifact_db_path("run_research_llm")
        client = SequenceLLMClient(
            [
                json.dumps(_policy_payload(), ensure_ascii=False),
                json.dumps(_impact_payload(), ensure_ascii=False),
                json.dumps(_company_payload(), ensure_ascii=False),
            ]
        )

        report = run_research(
            query="生成式人工智能服务提供者有哪些管理要求",
            db_path=db_path,
            ensure_sample_db=True,
            rebuild_sample_db=True,
            use_llm=True,
            llm_client=client,
        )

        self.assertEqual(len(client.calls), 3)
        self.assertIn("PolicyChain 政策研究报告", report)
        self.assertIn("清源模型安全科技", report)
        for term in PROHIBITED_TERMS:
            self.assertNotIn(term, report)

    def test_run_research_uses_resolved_default_database(self) -> None:
        db_path = artifact_db_path("run_research_resolved_default")
        ingest_sample_database(db_path=db_path, reset=True)

        with patch("scripts.run_research.resolve_default_db_path", return_value=db_path):
            report = run_research(
                query="生成式人工智能服务提供者有哪些管理要求",
                ensure_sample_db=False,
            )

        self.assertIn("PolicyChain 政策研究报告", report)
        self.assertIn("生成式人工智能服务管理暂行办法", report)

    def test_run_research_does_not_create_sample_database_at_full_db_path(self) -> None:
        full_db_path = artifact_db_path("missing_full_policy_db")

        with patch(
            "scripts.run_research.is_full_db_path",
            side_effect=lambda path: Path(path).resolve() == full_db_path.resolve(),
        ):
            with self.assertRaises(FileNotFoundError):
                run_research(
                    query="生成式人工智能服务提供者有哪些管理要求",
                    db_path=full_db_path,
                    ensure_sample_db=True,
                )

        self.assertFalse(full_db_path.exists())

    def test_cli_database_selection_flags(self) -> None:
        self.assertEqual(_selected_db_path(parse_args(["--sample-db"])), SAMPLE_DB_PATH)
        self.assertEqual(_selected_db_path(parse_args(["--full-db"])), FULL_DB_PATH)

        stderr = io.StringIO()
        with self.assertRaises(SystemExit):
            with redirect_stderr(stderr):
                parse_args(["--sample-db", "--full-db"])

    def test_parse_args_accepts_mcp_flags(self) -> None:
        args = parse_args(
            [
                "--mcp",
                "--mcp-config",
                ".mcp.example.json",
                "--mcp-timeout",
                "12",
                "--no-mcp-cache",
                "--skip-annual-reports",
            ]
        )

        self.assertTrue(args.mcp)
        self.assertEqual(args.mcp_config, ".mcp.example.json")
        self.assertEqual(args.mcp_timeout, 12)
        self.assertTrue(args.no_mcp_cache)
        self.assertTrue(args.skip_annual_reports)


class SequenceLLMClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if not self.responses:
            raise AssertionError("No LLM response left")
        return self.responses.pop(0)


def _policy_payload() -> dict[str, object]:
    return {
        "policy_identity": {
            "policy_id": "POL-2023-NAT-0048",
            "title": "生成式人工智能服务管理暂行办法",
            "document_number": "第15号",
            "publish_date": "2023-05-23",
            "issuing_agencies": ["国家级政府或主管部门（按官方来源）"],
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
        "implementation_mechanisms": ["算法模型治理"],
        "implementation_chain": [
            {
                "step_index": 1,
                "actor": "生成式人工智能服务提供者",
                "action": "落实模型、算法和训练数据治理",
                "mechanism": "算法模型治理",
                "evidence": [_evidence()],
            }
        ],
        "industry_impacts": [
            {
                "industry": "算法模型研发与评估",
                "impact_type": "direct",
                "direction": "mixed",
                "transmission_logic": "政策要求模型、算法和训练数据环节承担安全治理责任。",
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
                "company_name": "清源模型安全科技",
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
                "conditions": ["需核验真实官网、年报和公告资料。"],
                "risks": ["本地 mock 数据不可代表真实公司资料。"],
                "data_date": "2026-01-15",
                "confidence": 0.86,
            }
        ],
        "uncertainties": ["公司资料来自本地 mock 数据，仅用于验证流程。"],
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
