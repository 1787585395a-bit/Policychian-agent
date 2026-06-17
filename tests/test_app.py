from __future__ import annotations

import json
import os
import time
import unittest
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
    render_page,
    run_query,
)
from policychain.llm import LLMConfigurationError
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
        self.assertIn("/api/research", html)
        self.assertIn("/api/research-status", html)
        self.assertIn("use_llm: true", html)
        self.assertIn("use_mcp: true", html)
        self.assertEqual(len(EXAMPLE_QUERIES), 1)
        self.assertIn(EXAMPLE_QUERIES[0], html)

    def test_render_page_can_disable_default_mcp_for_cloud(self) -> None:
        with patch.dict(os.environ, {"POLICYCHAIN_ENABLE_MCP_BY_DEFAULT": "0"}):
            html = render_page().decode("utf-8")

        self.assertFalse(_mcp_default_enabled({"POLICYCHAIN_ENABLE_MCP_BY_DEFAULT": "0"}))
        self.assertIn("use_mcp: false", html)

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

    def test_run_query_returns_report(self) -> None:
        result = run_query(DEFAULT_POLICY_INPUT, db_path=artifact_db_path("app_query"))

        self.assertIn("PolicyChain", result["report"])
        self.assertGreaterEqual(result["elapsed_seconds"], 0)
        self.assertFalse(result["use_llm"])

    def test_run_query_can_request_llm_mode(self) -> None:
        with patch("app.run_research", return_value="# PolicyChain 政策研究报告") as fake_runner:
            result = run_query("测试政策正文", db_path=":memory:", use_llm=True)

        self.assertTrue(result["use_llm"])
        self.assertEqual(result["report"], "# PolicyChain 政策研究报告")
        self.assertTrue(fake_runner.call_args.kwargs["use_llm"])

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
        self.assertTrue(any(stage == "MCP 初始化" for _, stage, _ in progress_events))
        self.assertIsNone(fake_runner.call_args.kwargs["mcp_invoker"])

    def test_run_query_falls_back_when_llm_is_unconfigured(self) -> None:
        progress_events: list[tuple[int, str, str]] = []
        with patch(
            "app.run_research",
            side_effect=[
                LLMConfigurationError("DEEPSEEK_API_KEY is required"),
                "# PolicyChain 政策研究报告",
            ],
        ) as fake_runner:
            result = run_query(
                "测试政策正文",
                db_path=":memory:",
                use_llm=True,
                progress_callback=lambda progress, stage, message: progress_events.append((progress, stage, message)),
            )

        self.assertFalse(result["use_llm"])
        self.assertEqual(fake_runner.call_count, 2)
        self.assertIn("运行环境提示", result["report"])
        self.assertTrue(any(stage == "模型初始化" for _, stage, _ in progress_events))

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


    def test_vercel_wsgi_entrypoint_serves_health_and_research_api(self) -> None:
        from api.index import app as wsgi_app

        status, _, body = _call_wsgi(wsgi_app, "GET", "/healthz")

        self.assertEqual(status, "200 OK")
        self.assertEqual(json.loads(body.decode("utf-8"))["status"], "ok")

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
