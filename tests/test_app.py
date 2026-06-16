from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from app import DEFAULT_POLICY_INPUT, EXAMPLE_QUERIES, _create_job, _job_view, render_page, run_query
from tests.helpers import artifact_db_path


class AppTests(unittest.TestCase):
    def test_render_page_contains_policy_input_progress_and_logs(self) -> None:
        html = render_page().decode("utf-8")

        self.assertIn("<form", html)
        self.assertIn("粘贴政策链接或正文", html)
        self.assertIn("政策链接或政策正文", html)
        self.assertIn(DEFAULT_POLICY_INPUT, html)
        self.assertIn('name="use_llm"', html)
        self.assertIn("DeepSeek", html)
        self.assertIn('name="use_mcp"', html)
        self.assertIn("MCP 外部证据", html)
        self.assertNotIn('id="use_mcp" name="use_mcp" type="checkbox" value="1" disabled', html)
        self.assertIn("等待分析结果", html)
        self.assertIn("研究辅助，不构成投资建议", html)
        self.assertIn("数据库：", html)
        self.assertIn("运行模式：确定性流程", html)
        self.assertIn('data-loading-label="分析中"', html)
        self.assertIn('id="progress-bar"', html)
        self.assertIn('aria-label="运行日志窗口"', html)
        self.assertIn('id="log-panel"', html)
        self.assertIn('id="log-meta"', html)
        self.assertIn('id="copy-log"', html)
        self.assertIn('id="copy-log" type="button" disabled', html)
        self.assertIn("/api/research", html)
        self.assertIn("/api/research-status", html)
        for example_query in EXAMPLE_QUERIES:
            self.assertIn(example_query, html)

    def test_render_page_marks_llm_mode_checkbox(self) -> None:
        html = render_page(use_llm=True).decode("utf-8")

        self.assertIn('id="use_llm"', html)
        self.assertIn("checked", html)
        self.assertIn("运行模式：DeepSeek", html)
        self.assertIn("当前模式：DeepSeek", html)

    def test_render_page_marks_mcp_mode_checkbox(self) -> None:
        html = render_page(use_mcp=True).decode("utf-8")

        self.assertIn('id="use_mcp"', html)
        self.assertIn("运行模式：确定性流程 + MCP 外部证据", html)
        self.assertIn("当前模式：确定性流程 + MCP 外部证据", html)

    def test_render_page_renders_report_content(self) -> None:
        html = render_page(query="测试", report="# 标题\n\n## 章节\n\n- 条目").decode("utf-8")

        self.assertIn("<h1>标题</h1>", html)
        self.assertIn("<h2>章节</h2>", html)
        self.assertIn("<li>条目</li>", html)

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
            result = run_query("测试问题", db_path=":memory:", use_llm=True)

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
            result = run_query("测试问题", db_path=":memory:", use_mcp=True)

        self.assertTrue(result["use_mcp"])
        self.assertTrue(fake_invoker.closed)
        self.assertIs(fake_runner.call_args.kwargs["mcp_invoker"], fake_invoker)
        self.assertNotIn("skip_annual_reports", fake_runner.call_args.kwargs)

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

    def test_async_job_reports_error_status(self) -> None:
        with patch("app.run_query", side_effect=RuntimeError("读取失败")):
            job_id = _create_job("bad", use_llm=False, use_mcp=False)
            view = _wait_for_job(job_id, expected_status="error")

        self.assertEqual(view["status"], "error")
        self.assertEqual(view["job_id"], job_id)
        self.assertIn("读取失败", view["error"])
        self.assertTrue(any(item["stage"] == "错误" for item in view["logs"]))
        self.assertTrue(any("读取失败" in item["message"] for item in view["logs"]))


def _wait_for_job(job_id: str, expected_status: str = "done") -> dict[str, object]:
    deadline = time.time() + 5
    view = _job_view(job_id)
    while time.time() < deadline:
        view = _job_view(job_id)
        if view["status"] == expected_status:
            return view
        time.sleep(0.05)
    raise AssertionError(f"Job did not reach {expected_status}: {view}")


if __name__ == "__main__":
    unittest.main()
