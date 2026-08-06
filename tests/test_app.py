from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection
from http.server import HTTPServer
from io import BytesIO
from unittest.mock import patch

from app import (
    DEFAULT_POLICY_INPUT,
    EXAMPLE_QUERIES,
    _create_job,
    _health_payload,
    _job_view,
    _mcp_default_enabled,
    _resolve_host,
    _resolve_port,
    _sync_jobs_enabled,
    PolicyChainRequestHandler,
    render_example_report_page,
    render_page,
    run_query,
)
from policychain.observability import RunRecorder
from policychain.state import PolicyResearchState
from tests.helpers import artifact_db_path


class AppTests(unittest.TestCase):
    def test_cloud_runtime_env_helpers(self) -> None:
        self.assertEqual(_resolve_host({}), "127.0.0.1")
        self.assertEqual(_resolve_host({"POLICYCHAIN_HOST": "0.0.0.0"}), "0.0.0.0")
        self.assertEqual(_resolve_port({}), 8000)
        self.assertEqual(_resolve_port({"POLICYCHAIN_PORT": "8010"}), 8010)
        self.assertEqual(_resolve_port({"PORT": "10000", "POLICYCHAIN_PORT": "8010"}), 10000)
        self.assertFalse(_sync_jobs_enabled({}))
        self.assertTrue(_sync_jobs_enabled({"VERCEL": "1"}))
        self.assertTrue(_sync_jobs_enabled({"POLICYCHAIN_SYNC_JOBS": "true"}))

    def test_health_payload_is_render_friendly(self) -> None:
        payload = _health_payload()

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["service"], "policychain")
        self.assertIn("time", payload)

    def test_render_page_contains_policy_input_progress_and_logs(self) -> None:
        html = render_page().decode("utf-8")

        self.assertIn("<form", html)
        self.assertIn("粘贴政策链接或政策正文", html)
        self.assertIn("政策链接或政策正文", html)
        self.assertIn(DEFAULT_POLICY_INPUT, html)
        self.assertIn("默认启用模型分析与外部证据工具", html)
        self.assertNotIn("数据库：", html)
        self.assertNotIn("运行模式：", html)
        self.assertNotIn('name="use_llm"', html)
        self.assertNotIn('name="use_mcp"', html)
        self.assertNotIn("DeepSeek", html)
        self.assertNotIn("MCP 外部证据", html)
        self.assertIn("等待分析结果", html)
        self.assertIn("研究辅助，不构成投资建议", html)
        self.assertIn('data-loading-label="分析中"', html)
        self.assertIn('id="progress-bar"', html)
        self.assertIn('aria-label="运行日志窗口"', html)
        self.assertIn('id="log-panel"', html)
        self.assertIn('id="log-meta"', html)
        self.assertIn('id="copy-log"', html)
        self.assertIn('id="copy-log" type="button" disabled', html)
        self.assertIn('id="run-summary"', html)
        self.assertIn('id="run-id"', html)
        self.assertIn('id="run-mode"', html)
        self.assertIn('id="fallback-status"', html)
        self.assertIn('aria-label="Agent 状态"', html)
        self.assertIn("政策分析", html)
        self.assertIn("行业影响", html)
        self.assertIn("公司匹配", html)
        self.assertIn("报告生成", html)
        self.assertIn('id="download-log"', html)
        self.assertIn("/api/run-log?job_id=", html)
        self.assertIn("/api/research", html)
        self.assertIn("/api/research-status", html)
        self.assertIn("查看示例报告", html)
        self.assertIn('href="/example-report"', html)
        self.assertIn("use_llm: true", html)
        self.assertIn("use_mcp: true", html)
        self.assertEqual(len(EXAMPLE_QUERIES), 1)
        self.assertIn(EXAMPLE_QUERIES[0], html)

    def test_render_page_can_disable_default_mcp_for_cloud(self) -> None:
        with patch.dict(os.environ, {"POLICYCHAIN_ENABLE_MCP_BY_DEFAULT": "0"}):
            html = render_page().decode("utf-8")

        self.assertFalse(_mcp_default_enabled({"POLICYCHAIN_ENABLE_MCP_BY_DEFAULT": "0"}))
        self.assertIn("use_mcp: false", html)

    def test_head_requests_are_ready_for_hf_health_checks(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), PolicyChainRequestHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            for path, expected_content_type in (
                ("/", "text/html; charset=utf-8"),
                ("/healthz", "application/json; charset=utf-8"),
                ("/example-report", "text/html; charset=utf-8"),
            ):
                connection = HTTPConnection(host, port, timeout=5)
                try:
                    connection.request("HEAD", path)
                    response = connection.getresponse()
                    body = response.read()
                finally:
                    connection.close()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.getheader("Content-Type"), expected_content_type)
                self.assertEqual(body, b"")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_render_example_report_page_contains_completed_report(self) -> None:
        html = render_example_report_page().decode("utf-8")

        self.assertIn("示例报告", html)
        self.assertIn("生成式人工智能服务管理暂行办法", html)
        self.assertIn("不构成任何投资建议", html)
        self.assertIn("IMP-005：未成年人保护相关服务", html)
        self.assertIn("三六零（601360）", html)
        self.assertIn("安恒信息（688023）", html)
        self.assertIn('href="/"', html)

    def test_http_handler_serves_example_report_page(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), PolicyChainRequestHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            connection = HTTPConnection(host, port, timeout=5)
            try:
                connection.request("GET", "/example-report")
                response = connection.getresponse()
                body = response.read().decode("utf-8")
            finally:
                connection.close()

            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("Content-Type"), "text/html; charset=utf-8")
            self.assertIn("示例报告", body)
            self.assertIn("生成式人工智能服务管理暂行办法", body)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_render_page_renders_markdown_report_content(self) -> None:
        report = (
            "# 报告标题\n\n"
            "#### **一、政策核心与总体解读**\n\n"
            "1. **政策靶向更精准**：不同技术成熟度的产业面临不同矛盾。\n"
            "2. **资源配置逻辑重构**：政策工具分类施策。\n\n"
            "##### **路径一：未来产业示范与风险分担机制（IMP-001）**\n\n"
            "* **政策含义与措施**：建立投入增长和风险分担机制。\n"
            "- 影响集中在研发、示范应用及产业化初期。\n\n"
            "主体段落第一句。\n"
            "主体段落第二句。"
        )
        html = render_page(query="测试", report=report).decode("utf-8")

        self.assertIn("<h1>报告标题</h1>", html)
        self.assertIn("<h4><strong>一、政策核心与总体解读</strong></h4>", html)
        self.assertIn("<h5><strong>路径一：未来产业示范与风险分担机制（IMP-001）</strong></h5>", html)
        self.assertIn("<ol>", html)
        self.assertIn("<ul>", html)
        self.assertIn("<strong>政策靶向更精准</strong>", html)
        self.assertIn("<p>主体段落第一句。 主体段落第二句。</p>", html)

    def test_render_page_renders_error_content(self) -> None:
        html = render_page(query="测试", error="数据库不可用").decode("utf-8")

        self.assertIn("数据库不可用", html)
        self.assertIn('class="error"', html)

    def test_client_terminal_states_render_distinct_report_content(self) -> None:
        html = render_page().decode("utf-8")

        self.assertIn(
            "report.innerHTML = '<div class=\"empty\">正在分析，请等待结果。</div>';",
            html,
        )
        self.assertIn(
            "report.innerHTML = payload.report_html || '<div class=\"empty\">报告为空。</div>';",
            html,
        )
        show_error_script = html.split("function showError(message)", maxsplit=1)[1].split(
            "function escapeHtml(value)", maxsplit=1
        )[0]
        self.assertIn("分析失败，未生成报告。", show_error_script)
        self.assertNotIn("正在分析", show_error_script)

    def test_run_query_can_explicitly_use_deterministic_fallback(self) -> None:
        result = run_query(DEFAULT_POLICY_INPUT, db_path=artifact_db_path("app_query"), use_llm=False)

        self.assertIn("PolicyChain", result["report"])
        self.assertGreaterEqual(result["elapsed_seconds"], 0)
        self.assertFalse(result["use_llm"])

    def test_run_query_defaults_to_llm_mode(self) -> None:
        with patch("app.run_research", return_value="# PolicyChain 政策研究报告") as fake_runner:
            result = run_query("测试政策正文", db_path=":memory:")

        self.assertTrue(result["use_llm"])
        self.assertEqual(result["report"], "# PolicyChain 政策研究报告")
        self.assertTrue(fake_runner.call_args.kwargs["use_llm"])
        self.assertTrue(fake_runner.call_args.kwargs["return_state"])
        self.assertIsInstance(fake_runner.call_args.kwargs["run_recorder"], RunRecorder)

    def test_run_query_can_request_mcp_mode(self) -> None:
        class FakeClosableInvoker:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        fake_invoker = FakeClosableInvoker()
        with patch("app._build_mcp_invoker", return_value=fake_invoker), patch(
            "app.run_research",
            return_value="# PolicyChain 政策研究报告",
        ) as fake_runner:
            result = run_query("测试政策正文", db_path=":memory:", use_mcp=True)

        self.assertTrue(result["use_mcp"])
        self.assertTrue(fake_invoker.closed)
        self.assertIs(fake_runner.call_args.kwargs["mcp_invoker"], fake_invoker)
        self.assertNotIn("skip_annual_reports", fake_runner.call_args.kwargs)

    def test_run_query_logs_successful_mcp_initialization(self) -> None:
        class FakeClosableInvoker:
            def close(self) -> None:
                return None

        progress_events: list[tuple[int, str, str]] = []
        with patch("app._build_mcp_invoker", return_value=FakeClosableInvoker()), patch(
            "app.run_research",
            return_value="# PolicyChain report",
        ):
            run_query(
                "policy text",
                db_path=":memory:",
                use_mcp=True,
                progress_callback=lambda progress, stage, message: progress_events.append((progress, stage, message)),
            )

        self.assertTrue(any(stage == "MCP 初始化" for _, stage, _ in progress_events))
        self.assertTrue(any("Open-WebSearch/CNFinancial" in message for _, _, message in progress_events))

    def test_run_query_falls_back_when_mcp_config_is_missing(self) -> None:
        progress_events: list[tuple[int, str, str]] = []
        with patch("app._build_mcp_invoker", side_effect=FileNotFoundError(".mcp.local.json")), patch(
            "app.run_research",
            return_value="# PolicyChain 政策研究报告",
        ) as fake_runner:
            result = run_query(
                "测试政策正文",
                db_path=":memory:",
                use_mcp=True,
                progress_callback=lambda progress, stage, message: progress_events.append((progress, stage, message)),
            )

        self.assertFalse(result["use_mcp"])
        self.assertIn("运行环境提示", result["report"])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["requested_run_mode"], "llm")
        self.assertEqual(result["effective_run_mode"], "llm")
        self.assertTrue(any(stage == "MCP 初始化" for _, stage, _ in progress_events))
        self.assertIsNone(fake_runner.call_args.kwargs["mcp_invoker"])

    def test_run_query_falls_back_when_llm_is_unconfigured(self) -> None:
        progress_events: list[tuple[int, str, str]] = []
        recorder = RunRecorder(mode="llm")
        recorder.mark_fallback("workflow", "DEEPSEEK_API_KEY is required", "deterministic")
        recorder.mode = "deterministic"
        state = PolicyResearchState(
            user_query="测试政策正文",
            final_report="# PolicyChain 政策研究报告",
            run_id=recorder.run_id,
            run_mode="deterministic",
            fallback_used=True,
        )
        with patch("app.run_research", return_value=state) as fake_runner:
            result = run_query(
                "测试政策正文",
                db_path=":memory:",
                use_llm=True,
                run_recorder=recorder,
                progress_callback=lambda progress, stage, message: progress_events.append((progress, stage, message)),
            )

        self.assertFalse(result["use_llm"])
        self.assertEqual(fake_runner.call_count, 1)
        self.assertIn("PolicyChain", result["report"])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["requested_run_mode"], "llm")
        self.assertEqual(result["effective_run_mode"], "deterministic")
        self.assertIs(fake_runner.call_args.kwargs["run_recorder"], recorder)

    def test_async_job_reports_done_status_and_progress_logs(self) -> None:
        def fake_run_query(query: str, **kwargs):
            callback = kwargs["progress_callback"]
            callback(5, "读取用户输入政策", "正在读取政策链接或正文")
            callback(8, "URL 抓取", "输入为政策正文，跳过 URL 抓取")
            callback(12, "正文质量校验", "已读取 120 字政策正文")
            callback(15, "检索相似政策", "找到 0 条相似政策证据")
            callback(30, "政策分析", "已完成政策身份、措施和力度分析")
            callback(55, "行业影响分析", "已完成实施路径和行业影响分析")
            callback(75, "公司业务匹配", "已完成候选公司业务匹配")
            callback(90, "生成报告", "正在整合报告")
            callback(100, "完成", "报告已生成")
            return {
                "query": query,
                "report": "# PolicyChain 政策研究报告",
                "use_llm": False,
                "use_mcp": False,
                "elapsed_seconds": 0.1,
            }

        with patch("app.run_query", side_effect=fake_run_query):
            job_id = _create_job("政策正文", use_llm=False, use_mcp=False)
            view = _wait_for_job(job_id)

        self.assertEqual(view["status"], "done")
        self.assertEqual(view["job_id"], job_id)
        self.assertEqual(view["progress"], 100)
        self.assertIn("<h1>PolicyChain 政策研究报告</h1>", view["report_html"])
        stages = {item["stage"] for item in view["logs"]}
        self.assertIn("读取用户输入政策", stages)
        self.assertIn("URL 抓取", stages)
        self.assertIn("正文质量校验", stages)
        self.assertIn("政策分析", stages)
        self.assertIn("检索相似政策", stages)
        self.assertIn("行业影响分析", stages)
        self.assertIn("公司业务匹配", stages)
        self.assertIn("生成报告", stages)

    def test_sync_job_mode_finishes_before_returning_job_id(self) -> None:
        def fake_run_query(query: str, **kwargs):
            callback = kwargs["progress_callback"]
            callback(100, "完成", "报告已生成")
            return {
                "query": query,
                "report": "# PolicyChain 政策研究报告",
                "use_llm": False,
                "use_mcp": False,
                "elapsed_seconds": 0.1,
            }

        with patch.dict(os.environ, {"POLICYCHAIN_SYNC_JOBS": "1"}), patch("app.run_query", side_effect=fake_run_query):
            job_id = _create_job("政策正文", use_llm=False, use_mcp=False)
            view = _job_view(job_id)

        self.assertEqual(view["status"], "done")
        self.assertEqual(view["progress"], 100)
        self.assertIn("<h1>PolicyChain 政策研究报告</h1>", view["report_html"])

    def test_async_job_reports_error_status(self) -> None:
        with patch("app.run_query", side_effect=RuntimeError("读取失败")):
            job_id = _create_job("bad", use_llm=False, use_mcp=False)
            view = _wait_for_job(job_id, expected_status="error")

        self.assertEqual(view["status"], "error")
        self.assertEqual(view["job_id"], job_id)
        self.assertIn("读取失败", view["error"])
        self.assertTrue(any(item["stage"] == "错误" for item in view["logs"]))
        self.assertTrue(any("读取失败" in item["message"] for item in view["logs"]))

    def test_job_status_exposes_run_observability_without_serializing_recorder(self) -> None:
        def fake_run_query(query: str, **kwargs):
            recorder = kwargs["run_recorder"]
            recorder.set_agent_status("policy_analyst", "completed")
            recorder.set_agent_status("impact_analyst", "completed")
            recorder.set_agent_status("company_matcher", "completed")
            recorder.set_agent_status("report_writer", "completed")
            recorder.mark_fallback("company_matcher", "上游工具不可用", "deterministic_company_matcher")
            recorder.finish("completed")
            return {
                "query": query,
                "report": "# PolicyChain 政策研究报告",
                "use_llm": False,
                "use_mcp": False,
                "elapsed_seconds": 0.1,
                "run_id": recorder.run_id,
                "requested_run_mode": "llm",
                "effective_run_mode": "deterministic",
                "agent_status": recorder.agent_status,
                "fallback_used": True,
            }

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"POLICYCHAIN_SYNC_JOBS": "1", "POLICYCHAIN_RUN_LOG_DIR": temp_dir},
        ), patch("app.run_query", side_effect=fake_run_query):
            first_job_id = _create_job("政策正文一", use_llm=True, use_mcp=False)
            second_job_id = _create_job("政策正文二", use_llm=True, use_mcp=False)
            first = _job_view(first_job_id)
            second = _job_view(second_job_id)

            self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertEqual(first["requested_run_mode"], "llm")
            self.assertEqual(first["effective_run_mode"], "deterministic")
            self.assertEqual(
                first["agent_status"],
                {"policy": "completed", "impact": "completed", "company": "completed", "report": "completed"},
            )
            self.assertTrue(first["fallback_used"])
            self.assertTrue(first["log_download_available"])
            self.assertNotIn("run_recorder", first)
            json.dumps(first, ensure_ascii=False)

    def test_http_handler_downloads_redacted_logs_for_done_and_error_jobs(self) -> None:
        def successful_run_query(query: str, **kwargs):
            recorder = kwargs["run_recorder"]
            recorder.record(
                "tool.response",
                stage="company_matcher",
                status="ok",
                api_key="top-secret-value",
                content="不应出现在下载中的大段政策正文",
            )
            recorder.set_agent_status("company_matcher", "completed")
            recorder.finish("completed")
            return {
                "query": query,
                "report": "# 普通研究报告",
                "use_llm": False,
                "use_mcp": False,
                "elapsed_seconds": 0.1,
                "run_id": recorder.run_id,
                "requested_run_mode": "deterministic",
                "effective_run_mode": "deterministic",
                "agent_status": recorder.agent_status,
                "fallback_used": False,
            }

        def failed_run_query(query: str, **kwargs):
            recorder = kwargs["run_recorder"]
            recorder.record(
                "tool.error",
                stage="policy_analyst",
                status="failed",
                message="Authorization=top-secret-value",
            )
            raise RuntimeError("正文读取失败")

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"POLICYCHAIN_SYNC_JOBS": "1", "POLICYCHAIN_RUN_LOG_DIR": temp_dir},
        ):
            with patch("app.run_query", side_effect=successful_run_query):
                done_job_id = _create_job("政策正文", use_llm=False, use_mcp=False)
            with patch("app.run_query", side_effect=failed_run_query):
                error_job_id = _create_job("错误正文", use_llm=False, use_mcp=False)

            done_view = _job_view(done_job_id)
            error_view = _job_view(error_job_id)
            self.assertEqual(done_view["status"], "done")
            self.assertEqual(error_view["status"], "error")
            self.assertTrue(done_view["log_download_available"])
            self.assertTrue(error_view["log_download_available"])
            self.assertNotIn("tool.response", done_view["report_html"])

            server = HTTPServer(("127.0.0.1", 0), PolicyChainRequestHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                host, port = server.server_address
                for job_id, expected_status in ((done_job_id, "completed"), (error_job_id, "failed")):
                    connection = HTTPConnection(host, port, timeout=5)
                    try:
                        connection.request("GET", f"/api/run-log?job_id={job_id}")
                        response = connection.getresponse()
                        body = response.read()
                    finally:
                        connection.close()
                    decoded = body.decode("utf-8")
                    artifact = json.loads(decoded)
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.getheader("Content-Type"), "application/json; charset=utf-8")
                    self.assertIn("attachment; filename=", response.getheader("Content-Disposition") or "")
                    self.assertEqual(artifact["summary"]["status"], expected_status)
                    self.assertNotIn("top-secret-value", decoded)

                connection = HTTPConnection(host, port, timeout=5)
                try:
                    connection.request("GET", f"/api/run-logs/{done_view['run_id']}")
                    compatibility_response = connection.getresponse()
                    compatibility_body = compatibility_response.read()
                finally:
                    connection.close()
                self.assertEqual(compatibility_response.status, 200)
                self.assertEqual(json.loads(compatibility_body.decode("utf-8"))["summary"]["run_id"], done_view["run_id"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


    def test_vercel_wsgi_entrypoint_serves_health_and_research_api(self) -> None:
        from api.index import app as wsgi_app

        status, _, body = _call_wsgi(wsgi_app, "GET", "/healthz")

        self.assertEqual(status, "200 OK")
        self.assertEqual(json.loads(body.decode("utf-8"))["status"], "ok")

        status, _, body = _call_wsgi(wsgi_app, "GET", "/example-report")

        self.assertEqual(status, "200 OK")
        decoded = body.decode("utf-8")
        self.assertIn("示例报告", decoded)
        self.assertIn("生成式人工智能服务管理暂行办法", decoded)

        with patch("api.index._create_job", return_value="job-123"):
            status, _, body = _call_wsgi(
                wsgi_app,
                "POST",
                "/api/research",
                body=json.dumps({"query": "policy text"}).encode("utf-8"),
                content_type="application/json",
            )

        self.assertEqual(status, "202 Accepted")
        self.assertEqual(json.loads(body.decode("utf-8"))["job_id"], "job-123")


def _wait_for_job(job_id: str, expected_status: str = "done") -> dict[str, object]:
    deadline = time.time() + 5
    view = _job_view(job_id)
    while time.time() < deadline:
        view = _job_view(job_id)
        if view["status"] == expected_status:
            return view
        time.sleep(0.05)
    raise AssertionError(f"Job did not reach {expected_status}: {view}")


def _call_wsgi(
    wsgi_app,
    method: str,
    path: str,
    body: bytes = b"",
    content_type: str = "",
) -> tuple[str, list[tuple[str, str]], bytes]:
    captured: dict[str, object] = {}

    def start_response(status: str, headers: list[tuple[str, str]]) -> None:
        captured["status"] = status
        captured["headers"] = headers

    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "QUERY_STRING": "",
        "CONTENT_LENGTH": str(len(body)),
        "CONTENT_TYPE": content_type,
        "wsgi.input": BytesIO(body),
    }
    response_body = b"".join(wsgi_app(environ, start_response))
    return str(captured["status"]), list(captured["headers"]), response_body


if __name__ == "__main__":
    unittest.main()
