from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from api.index import app as api_app
from app import JOBS, JOBS_LOCK
from policychain.agents.llm_company_matcher import run_llm_company_matcher
from policychain.graph import run_policy_research_workflow
from policychain.llm import LLMConfigurationError
from policychain.mcp import FakeMCPInvoker
from policychain.observability import RunRecorder, load_run_artifact, record_event
from policychain.state import PolicyResearchState
from policychain.tools.mcp_tools import CNFINANCIAL_SERVER
from scripts.run_research import run_research
from tests.helpers import build_sample_store


class ObservabilityTests(unittest.TestCase):
    def test_recorder_writes_loadable_redacted_summary_and_events(self) -> None:
        with TemporaryDirectory() as directory:
            recorder = RunRecorder(log_root=directory, mode="llm")
            with recorder.activate():
                record_event(
                    "test.secret",
                    stage="test",
                    status="ok",
                    authorization="Bearer live-secret-token",
                    nested={"api_key": "sk-secret", "cookie": "session=secret"},
                    policy_text="完整政策正文不应默认写入日志",
                )
                record_event(
                    "llm.call.end",
                    stage="policy_analyst",
                    status="ok",
                    system_prompt="默认不保存完整系统提示词",
                    user_prompt="默认不保存完整用户提示词",
                    response="默认不保存完整响应",
                )
                recorder.set_agent_status("policy_analyst", "completed")
                recorder.mark_fallback("impact_analyst", "provider error", "deterministic_impact_analyst")
            recorder.finish("completed")

            artifact = load_run_artifact(recorder.run_id, log_root=directory)

            self.assertEqual(artifact["summary"]["run_id"], recorder.run_id)
            self.assertTrue(artifact["summary"]["fallback_used"])
            serialized = json.dumps(artifact, ensure_ascii=False)
            self.assertNotIn("live-secret-token", serialized)
            self.assertNotIn("sk-secret", serialized)
            self.assertNotIn("完整政策正文不应默认写入日志", serialized)
            secret_event = next(item for item in artifact["events"] if item["event_type"] == "test.secret")
            self.assertEqual(secret_event["authorization"], "[REDACTED]")
            self.assertTrue(secret_event["policy_text"]["omitted"])
            llm_event = next(item for item in artifact["events"] if item["event_type"] == "llm.call.end")
            self.assertEqual(set(llm_event["response"]), {"omitted", "char_count", "sha256"})

    def test_full_llm_io_flag_records_only_redacted_llm_content(self) -> None:
        with TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"POLICYCHAIN_LOG_FULL_LLM_IO": "1", "POLICYCHAIN_LOG_INCLUDE_CONTENT": "0"},
        ):
            recorder = RunRecorder(log_root=directory)
            with recorder.activate():
                record_event(
                    "llm.call.end",
                    stage="policy_analyst",
                    status="ok",
                    system_prompt="系统提示 Authorization=prompt-secret",
                    user_prompt="用户提示 Cookie=session-secret",
                    response="响应 api_key=response-secret",
                )
                record_event(
                    "workflow.input",
                    stage="policy_analyst",
                    status="ok",
                    policy_text="即使开启 LLM IO 也不单独记录完整政策正文",
                )
            recorder.finish("completed")

            artifact = load_run_artifact(recorder.run_id, log_root=directory)

        llm_event = next(item for item in artifact["events"] if item["event_type"] == "llm.call.end")
        workflow_event = next(item for item in artifact["events"] if item["event_type"] == "workflow.input")
        self.assertEqual(llm_event["system_prompt"], "系统提示 Authorization=[REDACTED]")
        self.assertEqual(llm_event["user_prompt"], "用户提示 Cookie=[REDACTED]")
        self.assertEqual(llm_event["response"], "响应 api_key=[REDACTED]")
        self.assertTrue(workflow_event["policy_text"]["omitted"])

    def test_logging_failure_is_fail_open(self) -> None:
        with TemporaryDirectory() as directory:
            blocking_file = Path(directory) / "not-a-directory"
            blocking_file.write_text("block", encoding="utf-8")

            recorder = RunRecorder(log_root=blocking_file)
            with recorder.activate():
                record_event("test.event", status="ok")
            summary = recorder.finish("completed")

            self.assertEqual(summary["status"], "completed")
            self.assertTrue(recorder.logging_errors)

    def test_workflow_exposes_run_id_and_agent_rule_mcp_report_events(self) -> None:
        store = build_sample_store()
        try:
            with TemporaryDirectory() as directory:
                recorder = RunRecorder(log_root=directory)
                state = run_policy_research_workflow(
                    "生成式人工智能服务提供者有哪些管理要求",
                    store,
                    run_recorder=recorder,
                )
                artifact = load_run_artifact(state.run_id, log_root=directory)

            event_types = {item["event_type"] for item in artifact["events"]}
            self.assertEqual(state.run_id, recorder.run_id)
            self.assertEqual(state.agent_status["report_writer"], "completed")
            self.assertIn("agent.status", event_types)
            self.assertIn("mcp.call", event_types)
            self.assertIn("candidate.pipeline", event_types)
            self.assertIn("report.source", event_types)
        finally:
            store.close()

    def test_real_run_research_configuration_fallback_has_one_completed_finish(self) -> None:
        with TemporaryDirectory() as directory:
            recorder = RunRecorder(log_root=directory, mode="llm")
            with patch(
                "policychain.graph.create_llm_client",
                side_effect=LLMConfigurationError("DEEPSEEK_API_KEY is required"),
            ):
                state = run_research(
                    query="生成式人工智能服务提供者有哪些管理要求",
                    db_path=Path(directory) / "fallback.sqlite",
                    use_llm=True,
                    run_recorder=recorder,
                    return_state=True,
                )
            artifact = load_run_artifact(recorder.run_id, log_root=directory)

        finish_events = [item for item in artifact["events"] if item["event_type"] == "run.finish"]
        self.assertEqual(state.run_id, recorder.run_id)
        self.assertEqual(state.run_mode, "deterministic")
        self.assertTrue(state.fallback_used)
        self.assertEqual(len(finish_events), 1)
        self.assertEqual(finish_events[0]["status"], "completed")
        self.assertEqual(artifact["summary"]["status"], "completed")
        self.assertTrue(artifact["summary"]["fallback_used"])
        self.assertEqual(artifact["summary"]["agent_status"]["report_writer"], "completed")

    def test_react_candidate_correlates_original_tool_call_id_across_events_and_provenance(self) -> None:
        industry_call_count = 0

        def industry_stocks(**_kwargs):
            nonlocal industry_call_count
            industry_call_count += 1
            if industry_call_count == 1:
                return []
            return [{"名称": "电池公司", "代码": "300002", "主营业务": "动力电池制造"}]

        invoker = FakeMCPInvoker(
            {
                (CNFINANCIAL_SERVER, "get_industry_list"): [{"名称": "动力电池"}],
                (CNFINANCIAL_SERVER, "get_concept_list"): [],
                (CNFINANCIAL_SERVER, "get_industry_stocks"): industry_stocks,
                (CNFINANCIAL_SERVER, "search_stock"): [],
                (CNFINANCIAL_SERVER, "get_company_profile"): [{"主营业务": "动力电池制造与储能系统"}],
            }
        )
        client = _SequenceClient(
            [
                json.dumps({"thought": "证据足够", "action": "finish", "arguments": {}}, ensure_ascii=False),
                json.dumps(_company_match_payload(), ensure_ascii=False),
            ]
        )
        state = PolicyResearchState(
            user_query="动力电池政策",
            industry_impacts=[
                {
                    "impact_id": "IMP-001",
                    "industry": "动力电池",
                    "chain_segment": "动力电池制造",
                    "transmission_logic": "新能源汽车需求带动动力电池制造",
                    "business_variables": ["电池装机量"],
                    "affected_company_types": ["动力电池制造商"],
                    "conditions": [],
                    "risks": [],
                }
            ],
        )
        with TemporaryDirectory() as directory, patch.dict(os.environ, {"POLICYCHAIN_MCP_FAST_MODE": "1"}):
            recorder = RunRecorder(log_root=directory, mode="llm")
            with recorder.activate():
                run_llm_company_matcher(state, llm_client=client, mcp_invoker=invoker)
            recorder.finish("completed")
            artifact = load_run_artifact(recorder.run_id, log_root=directory)

        accepted = next(item for item in state.react_candidate_audit if item["decision"] == "accept")
        tool_call_id = accepted["tool_call_id"]
        candidate = next(item for item in state.company_candidates if item["stock_code"] == "300002")
        provenance = next(item for item in candidate["provenance"] if item["source_type"] == "cnfinancial_react")
        mcp_event = next(
            item
            for item in artifact["events"]
            if item["event_type"] == "mcp.call"
            and item.get("status") == "result"
            and item.get("tool_call_id") == tool_call_id
        )
        react_event = next(
            item
            for item in artifact["events"]
            if item["event_type"] == "react.candidate" and item.get("tool_call_id") == tool_call_id
        )
        self.assertTrue(tool_call_id.startswith("tool-"))
        self.assertEqual(provenance["tool_call_id"], tool_call_id)
        for event in (mcp_event, react_event):
            self.assertEqual(event["run_id"], recorder.run_id)
            self.assertEqual(event["stage"], "company_matcher")
            self.assertEqual(event["impact_id"], "IMP-001")
            self.assertEqual(event["company_name"], "电池公司")
            self.assertEqual(event["stock_code"], "300002")

    def test_wsgi_downloads_success_or_failure_artifact_by_run_id(self) -> None:
        with TemporaryDirectory() as directory:
            recorder = RunRecorder(log_root=directory)
            recorder.finish("failed", error="synthetic failure")
            with patch.dict(os.environ, {"POLICYCHAIN_RUN_LOG_DIR": directory}):
                status, headers, body = _call_wsgi(f"/api/run-logs/{recorder.run_id}")

            payload = json.loads(body.decode("utf-8"))
            self.assertEqual(status, "200 OK")
            self.assertEqual(payload["summary"]["status"], "failed")
            self.assertIn("attachment", headers["Content-Disposition"])
            self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")

    def test_wsgi_run_log_query_string_matches_app_job_download_contract(self) -> None:
        job_ids = ["job-observability-success", "job-observability-failed"]
        with TemporaryDirectory() as directory:
            completed = RunRecorder(log_root=directory)
            completed.finish("completed")
            failed = RunRecorder(log_root=directory)
            failed.finish("failed", error="synthetic failure")
            with JOBS_LOCK:
                JOBS[job_ids[0]] = {"run_id": completed.run_id, "run_recorder": completed, "status": "done"}
                JOBS[job_ids[1]] = {"run_id": failed.run_id, "run_recorder": failed, "status": "error"}
            try:
                for job_id, expected_status in zip(job_ids, ("completed", "failed"), strict=True):
                    status, headers, body = _call_wsgi("/api/run-log", query_string=f"job_id={job_id}")
                    payload = json.loads(body.decode("utf-8"))
                    self.assertEqual(status, "200 OK")
                    self.assertEqual(payload["summary"]["status"], expected_status)
                    self.assertIn(payload["summary"]["run_id"], headers["Content-Disposition"])

                status, headers, body = _call_wsgi("/api/run-log", query_string="job_id=missing-job")
                self.assertEqual(status, "404 Not Found")
                self.assertEqual(json.loads(body.decode("utf-8"))["error"], "任务不存在")
                self.assertIn("policychain-run-log.json", headers["Content-Disposition"])
            finally:
                with JOBS_LOCK:
                    for job_id in job_ids:
                        JOBS.pop(job_id, None)


def _call_wsgi(path: str, *, query_string: str = "") -> tuple[str, dict[str, str], bytes]:
    captured: dict[str, object] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = dict(headers)

    body = b"".join(
        api_app(
            {
                "REQUEST_METHOD": "GET",
                "PATH_INFO": path,
                "QUERY_STRING": query_string,
                "CONTENT_LENGTH": "0",
                "wsgi.input": BytesIO(),
            },
            start_response,
        )
    )
    return str(captured["status"]), dict(captured["headers"]), body


class _SequenceClient:
    model = "test-sequence"

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self.responses:
            raise AssertionError("No response left")
        return self.responses.pop(0)


def _company_match_payload() -> dict[str, object]:
    return {
        "companies": [
            {
                "company_name": "电池公司",
                "stock_code": "300002",
                "impact_id": "IMP-001",
                "industry_segment": "动力电池",
                "matched_business": "动力电池制造与储能系统",
                "match_level": "high",
                "business_evidence": [
                    {
                        "source_name": "CNFinancial MCP",
                        "source_url": None,
                        "text": "公司主营动力电池制造与储能系统。",
                        "data_date": "2026-07-21",
                    }
                ],
                "policy_link": "新能源汽车需求带动动力电池制造。",
                "revenue_relevance": "unknown",
                "conditions": [],
                "risks": [],
                "data_date": "2026-07-21",
                "confidence": 0.85,
            }
        ],
        "uncertainties": [],
    }


if __name__ == "__main__":
    unittest.main()
