from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch
import unittest
from urllib.error import HTTPError
from uuid import uuid4

from policychain.llm import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    DeepSeekClient,
    LLMConfigurationError,
    LLMGeneration,
    LLMProviderError,
    MockLLMClient,
    create_llm_client,
    load_local_env,
)
from policychain.safety import SafetyViolation


class LLMTests(unittest.TestCase):
    def test_mock_llm_returns_deterministic_response(self) -> None:
        client = MockLLMClient(response="结构化政策分析结果")

        self.assertEqual(client.generate("system", "user"), "结构化政策分析结果")

    def test_mock_llm_requires_non_empty_prompts(self) -> None:
        client = MockLLMClient()

        with self.assertRaisesRegex(ValueError, "system_prompt"):
            client.generate("", "user")
        with self.assertRaisesRegex(ValueError, "user_prompt"):
            client.generate("system", "")

    def test_mock_llm_validates_response_safety(self) -> None:
        client = MockLLMClient(response="输出目标价")

        with self.assertRaises(SafetyViolation):
            client.generate("system", "user")

    def test_mock_llm_generation_metadata(self) -> None:
        client = MockLLMClient(response="安全输出", model="mock-v1")
        generation = client.generate_with_metadata("system", "user")

        self.assertIsInstance(generation, LLMGeneration)
        self.assertEqual(generation.text, "安全输出")
        self.assertEqual(generation.model, "mock-v1")
        self.assertEqual(generation.provider, "mock")

    def test_deepseek_client_builds_official_chat_completion_request(self) -> None:
        calls = []

        def fake_http_post(request, timeout):
            calls.append((request, timeout))
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "政策分析输出",
                            }
                        }
                    ]
                }
            )

        client = DeepSeekClient(
            api_key="test-key",
            model="deepseek-v4-pro",
            thinking_type="enabled",
            reasoning_effort="high",
            max_tokens=1024,
            http_post=fake_http_post,
        )

        self.assertEqual(client.generate("system", "user"), "政策分析输出")

        request, timeout = calls[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, f"{DEFAULT_DEEPSEEK_BASE_URL}/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(timeout, 60.0)
        self.assertEqual(body["model"], "deepseek-v4-pro")
        self.assertEqual(body["messages"][0], {"role": "system", "content": "system"})
        self.assertEqual(body["messages"][1], {"role": "user", "content": "user"})
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertEqual(body["max_tokens"], 1024)
        self.assertFalse(body["stream"])

    def test_deepseek_client_validates_response_safety(self) -> None:
        client = DeepSeekClient(
            api_key="test-key",
            http_post=lambda request, timeout: _FakeResponse(
                {"choices": [{"message": {"content": "输出目标价"}}]}
            ),
        )

        with self.assertRaises(SafetyViolation):
            client.generate("system", "user")

    def test_deepseek_client_rejects_malformed_response(self) -> None:
        client = DeepSeekClient(
            api_key="test-key",
            http_post=lambda request, timeout: _FakeResponse({"choices": []}),
        )

        with self.assertRaisesRegex(LLMProviderError, "missing choices"):
            client.generate("system", "user")

    def test_deepseek_client_wraps_http_errors(self) -> None:
        def fake_http_post(request, timeout):
            raise HTTPError(
                request.full_url,
                401,
                "Unauthorized",
                hdrs={},
                fp=_FakeErrorBody(b'{"error":"bad key"}'),
            )

        client = DeepSeekClient(api_key="test-key", http_post=fake_http_post)

        with self.assertRaisesRegex(LLMProviderError, "HTTP 401"):
            client.generate("system", "user")

    def test_deepseek_client_wraps_read_timeout(self) -> None:
        def fake_http_post(request, timeout):
            raise TimeoutError("The read operation timed out")

        client = DeepSeekClient(api_key="test-key", timeout_seconds=12, http_post=fake_http_post)

        with self.assertRaisesRegex(LLMProviderError, "timed out after 12"):
            client.generate("system", "user")

    def test_deepseek_client_from_env_and_factory(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "POLICYCHAIN_LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "env-key",
                "DEEPSEEK_MODEL": "deepseek-v4-pro",
                "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
                "DEEPSEEK_THINKING": "enabled",
                "DEEPSEEK_REASONING_EFFORT": "max",
                "DEEPSEEK_MAX_TOKENS": "2048",
            },
            clear=False,
        ):
            client = create_llm_client()

        self.assertIsInstance(client, DeepSeekClient)
        self.assertEqual(client.api_key, "env-key")
        self.assertEqual(client.model, "deepseek-v4-pro")
        self.assertEqual(client.base_url, "https://api.deepseek.com")
        self.assertEqual(client.thinking_type, "enabled")
        self.assertEqual(client.reasoning_effort, "max")
        self.assertEqual(client.max_tokens, 2048)
        self.assertFalse(client.use_system_proxy)

    def test_deepseek_client_can_opt_into_system_proxy(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "DEEPSEEK_API_KEY": "env-key",
                "DEEPSEEK_USE_SYSTEM_PROXY": "1",
            },
            clear=True,
        ):
            client = DeepSeekClient.from_env()

        self.assertTrue(client.use_system_proxy)

    def test_create_llm_client_defaults_to_mock(self) -> None:
        with patch.dict("os.environ", {"POLICYCHAIN_DISABLE_DOTENV": "1"}, clear=True):
            self.assertIsInstance(create_llm_client(), MockLLMClient)

    def test_create_llm_client_can_read_local_env_file(self) -> None:
        temp_dir = _workspace_temp_dir()
        env_file = temp_dir / ".env.local"
        env_file.write_text(
            "\n".join(
                [
                    "POLICYCHAIN_LLM_PROVIDER=deepseek",
                    "DEEPSEEK_API_KEY=local-key",
                    "DEEPSEEK_MODEL=deepseek-v4-pro",
                ]
            ),
            encoding="utf-8",
        )
        current_directory = os.getcwd()
        try:
            os.chdir(temp_dir)
            with patch.dict("os.environ", {}, clear=True):
                client = create_llm_client()
        finally:
            os.chdir(current_directory)

        self.assertIsInstance(client, DeepSeekClient)
        self.assertEqual(client.api_key, "local-key")
        self.assertEqual(client.model, "deepseek-v4-pro")

    def test_local_env_does_not_override_existing_environment(self) -> None:
        temp_dir = _workspace_temp_dir()
        env_file = temp_dir / ".env.local"
        env_file.write_text("DEEPSEEK_API_KEY=local-key", encoding="utf-8")
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "existing-key"}, clear=True):
            load_local_env(env_file)
            self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "existing-key")

    def test_deepseek_client_requires_api_key(self) -> None:
        with self.assertRaisesRegex(LLMConfigurationError, "DEEPSEEK_API_KEY"):
            DeepSeekClient(api_key="")

    def test_default_deepseek_model_uses_current_v4_name(self) -> None:
        self.assertEqual(DEFAULT_DEEPSEEK_MODEL, "deepseek-v4-flash")


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class _FakeErrorBody:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def close(self) -> None:
        return None


def _workspace_temp_dir() -> Path:
    path = Path("artifacts/test-results") / f"llm_env_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    unittest.main()
