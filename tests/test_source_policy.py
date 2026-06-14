from __future__ import annotations

import unittest
from unittest.mock import patch

from policychain.source_policy import (
    SourcePolicyError,
    build_source_policy_from_text,
    build_source_policy_from_url,
)


POLICY_TEXT = """生成式人工智能服务管理办法

第一条 为了促进生成式人工智能健康发展和规范应用，维护国家安全和社会公共利益，制定本办法。
第二条 提供生成式人工智能服务，应当依法承担网络信息内容生产者责任，履行网络信息安全义务。
第三条 服务提供者应当开展安全评估，采取有效措施防范违法和不良信息生成传播。
"""


class SourcePolicyTests(unittest.TestCase):
    def test_text_input_generates_stable_input_policy_and_chunks(self) -> None:
        first = build_source_policy_from_text(POLICY_TEXT, raw_input=POLICY_TEXT, input_type="text")
        second = build_source_policy_from_text(POLICY_TEXT, raw_input=POLICY_TEXT, input_type="text")

        self.assertTrue(first["policy_id"].startswith("INPUT-"))
        self.assertEqual(first["policy_id"], second["policy_id"])
        self.assertEqual(first["metadata"]["policy_id"], first["policy_id"])
        self.assertGreaterEqual(len(first["chunks"]), 1)
        self.assertTrue(first["chunks"][0]["chunk_id"].startswith(first["policy_id"]))
        self.assertIn("生成式人工智能服务管理办法", first["title"])

    def test_html_url_extracts_title_and_policy_text(self) -> None:
        html = """
        <html><head><title>政策页面标题</title><script>ignore()</script></head>
        <body><article><h1>生成式人工智能服务管理办法</h1>
        <p>第一条 为了促进生成式人工智能健康发展和规范应用，维护国家安全和社会公共利益，制定本办法。</p>
        <p>第二条 服务提供者应当履行安全义务，开展算法模型安全评估，提升训练数据质量。</p>
        <p>第三条 主管部门应当加强监督管理，推动行业组织建立服务能力评估机制。</p>
        </article></body></html>
        """

        with patch("policychain.source_policy.urlopen", return_value=_FakeResponse(html.encode("utf-8"), "text/html; charset=utf-8")):
            result = build_source_policy_from_url("https://example.test/policy.html")

        self.assertEqual(result["input_type"], "url")
        self.assertEqual(result["source_url"], "https://example.test/policy.html")
        self.assertEqual(result["title"], "生成式人工智能服务管理办法")
        self.assertIn("服务提供者应当履行安全义务", result["text"])

    def test_pdf_url_extracts_text(self) -> None:
        with patch("policychain.source_policy.urlopen", return_value=_FakeResponse(b"%PDF", "application/pdf")), patch(
            "policychain.source_policy.PdfReader",
            return_value=_FakePdfReader(),
        ):
            result = build_source_policy_from_url("https://example.test/policy.pdf")

        self.assertEqual(result["input_type"], "url")
        self.assertIn("服务提供者应当开展安全评估", result["text"])

    def test_url_fetch_failure_raises_clear_error(self) -> None:
        with patch("policychain.source_policy.urlopen", side_effect=OSError("network refused")):
            with self.assertRaisesRegex(SourcePolicyError, "Failed to fetch policy URL"):
                build_source_policy_from_url("https://example.test/missing")

    def test_non_policy_html_fails_quality_check(self) -> None:
        html = """
        <html><head><title>首页</title></head>
        <body><nav>产品 文档 登录</nav><main>欢迎使用系统。这里是导航页，不包含正式政策正文。</main></body></html>
        """

        with patch("policychain.source_policy.urlopen", return_value=_FakeResponse(html.encode("utf-8"), "text/html; charset=utf-8")):
            with self.assertRaisesRegex(SourcePolicyError, "正文质量校验失败"):
                build_source_policy_from_url("https://example.test/")

    def test_error_page_html_fails_quality_check(self) -> None:
        html = """
        <html><head><title>404 Not Found</title></head>
        <body><main>404 页面不存在，请登录后重试。Not Found.</main></body></html>
        """

        with patch("policychain.source_policy.urlopen", return_value=_FakeResponse(html.encode("utf-8"), "text/html; charset=utf-8")):
            with self.assertRaisesRegex(SourcePolicyError, "正文质量校验失败"):
                build_source_policy_from_url("https://example.test/not-found")


class _FakeResponse:
    def __init__(self, payload: bytes, content_type: str) -> None:
        self.payload = payload
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class _FakePdfPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakePdfReader:
    pages = [
        _FakePdfPage(
            "生成式人工智能服务管理办法\n"
            "第一条 为了促进生成式人工智能健康发展和规范应用，制定本办法。\n"
            "第二条 服务提供者应当开展安全评估，提升训练数据质量。\n"
            "第三条 主管部门应当加强监督管理，推动行业组织建立评估机制。"
        )
    ]


if __name__ == "__main__":
    unittest.main()
