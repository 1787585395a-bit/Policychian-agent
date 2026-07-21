from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
from time import perf_counter
from typing import Any, Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from policychain.safety import assert_no_investment_advice
from policychain.observability import record_event


DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
LOCAL_ENV_PATH = Path(".env.local")


class LLMClient(Protocol):
    """Minimal boundary for future real LLM providers."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text from a system and user prompt."""


@dataclass(frozen=True)
class LLMGeneration:
    text: str
    model: str
    provider: str


class LLMConfigurationError(ValueError):
    """Raised when an LLM provider is missing required configuration."""


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider call fails or returns malformed data."""


class MockLLMClient:
    """Deterministic no-network LLM stand-in for tests and local scaffolding."""

    def __init__(self, response: str = "mock llm response", model: str = "mock-policychain") -> None:
        self.response = response
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not system_prompt.strip():
            raise ValueError("system_prompt is required")
        if not user_prompt.strip():
            raise ValueError("user_prompt is required")
        assert_no_investment_advice(self.response, context="LLM response")
        return self.response

    def generate_with_metadata(self, system_prompt: str, user_prompt: str) -> LLMGeneration:
        return LLMGeneration(
            text=self.generate(system_prompt, user_prompt),
            model=self.model,
            provider="mock",
        )


class DeepSeekClient:
    """DeepSeek Chat Completions client using only the Python standard library."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_DEEPSEEK_MODEL,
        base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
        timeout_seconds: float = 60.0,
        thinking_type: str | None = "disabled",
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        use_system_proxy: bool = False,
        http_post: Callable[..., Any] | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMConfigurationError("DEEPSEEK_API_KEY is required")
        if not model.strip():
            raise LLMConfigurationError("DeepSeek model is required")
        if not base_url.strip():
            raise LLMConfigurationError("DeepSeek base_url is required")
        if thinking_type not in {None, "enabled", "disabled"}:
            raise LLMConfigurationError("thinking_type must be 'enabled', 'disabled', or None")
        if reasoning_effort not in {None, "high", "max"}:
            raise LLMConfigurationError("reasoning_effort must be 'high', 'max', or None")
        if max_tokens is not None and max_tokens <= 0:
            raise LLMConfigurationError("max_tokens must be positive")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.thinking_type = thinking_type
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.use_system_proxy = use_system_proxy
        self._http_post = http_post or (urlopen if use_system_proxy else _open_without_proxy)

    @classmethod
    def from_env(cls) -> "DeepSeekClient":
        load_local_env()
        timeout = float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "60"))
        max_tokens = _optional_int(os.getenv("DEEPSEEK_MAX_TOKENS"))
        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            model=os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
            base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL),
            timeout_seconds=timeout,
            thinking_type=os.getenv("DEEPSEEK_THINKING", "disabled"),
            reasoning_effort=os.getenv("DEEPSEEK_REASONING_EFFORT") or None,
            max_tokens=max_tokens,
            use_system_proxy=_truthy_env(os.getenv("DEEPSEEK_USE_SYSTEM_PROXY")),
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not system_prompt.strip():
            raise ValueError("system_prompt is required")
        if not user_prompt.strip():
            raise ValueError("user_prompt is required")

        request = self._build_request(system_prompt=system_prompt, user_prompt=user_prompt)
        try:
            with self._http_post(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = _read_error_body(exc)
            raise LLMProviderError(f"DeepSeek API returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            proxy_hint = (
                " Check DEEPSEEK_USE_SYSTEM_PROXY or local proxy settings."
                if self.use_system_proxy
                else ""
            )
            raise LLMProviderError(f"DeepSeek API request failed for {self.base_url}: {exc.reason}.{proxy_hint}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise LLMProviderError(
                f"DeepSeek API request timed out after {self.timeout_seconds} seconds"
            ) from exc
        except OSError as exc:
            raise LLMProviderError(f"DeepSeek API request failed for {self.base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LLMProviderError("DeepSeek API returned invalid JSON") from exc

        text = _extract_chat_completion_text(response_payload)
        assert_no_investment_advice(text, context="DeepSeek response")
        return text

    def generate_with_metadata(self, system_prompt: str, user_prompt: str) -> LLMGeneration:
        return LLMGeneration(
            text=self.generate(system_prompt, user_prompt),
            model=self.model,
            provider="deepseek",
        )

    def _build_request(self, system_prompt: str, user_prompt: str) -> Request:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        if self.thinking_type is not None:
            body["thinking"] = {"type": self.thinking_type}
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens

        return Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )


def create_llm_client(provider: str | None = None) -> LLMClient:
    load_local_env()
    selected_provider = (provider or os.getenv("POLICYCHAIN_LLM_PROVIDER") or "deepseek").strip().lower()
    if selected_provider == "mock":
        return MockLLMClient()
    if selected_provider == "deepseek":
        return DeepSeekClient.from_env()
    raise LLMConfigurationError(f"Unsupported LLM provider: {selected_provider}")


def observed_llm_generate(
    client: LLMClient,
    system_prompt: str,
    user_prompt: str,
    *,
    agent: str,
    prompt_version: str = "v1",
) -> str:
    provider = _llm_provider_name(client)
    model = str(getattr(client, "model", client.__class__.__name__))
    started = perf_counter()
    record_event(
        "llm.call.start",
        stage=agent,
        status="running",
        agent=agent,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    try:
        response = client.generate(system_prompt, user_prompt)
    except Exception as exc:
        record_event(
            "llm.call.end",
            stage=agent,
            status="error",
            agent=agent,
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            duration_ms=round((perf_counter() - started) * 1000, 3),
            error=f"{exc.__class__.__name__}: {str(exc)[:300]}",
        )
        raise
    record_event(
        "llm.call.end",
        stage=agent,
        status="ok",
        agent=agent,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        duration_ms=round((perf_counter() - started) * 1000, 3),
        response=response,
    )
    return response


def _llm_provider_name(client: LLMClient) -> str:
    name = client.__class__.__name__.lower()
    if "deepseek" in name:
        return "deepseek"
    if "mock" in name:
        return "mock"
    return client.__class__.__name__


def load_local_env(path: str | Path = LOCAL_ENV_PATH) -> None:
    """Load local KEY=VALUE settings without overriding existing environment variables."""

    if os.getenv("POLICYCHAIN_DISABLE_DOTENV") == "1":
        return

    env_path = Path(path)
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, _unquote_env_value(value.strip()))


def _extract_chat_completion_text(payload: dict[str, Any]) -> str:
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMProviderError("DeepSeek API response missing choices[0].message.content") from exc
    if not isinstance(text, str) or not text.strip():
        raise LLMProviderError("DeepSeek API response content is empty")
    return text


def _read_error_body(error: HTTPError) -> str:
    try:
        body = error.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    return body or error.reason


def _optional_int(raw_value: str | None) -> int | None:
    if raw_value is None or not raw_value.strip():
        return None
    try:
        return int(raw_value)
    except ValueError as exc:
        raise LLMConfigurationError(f"Invalid integer value: {raw_value}") from exc


def _truthy_env(raw_value: str | None) -> bool:
    return (raw_value or "").strip().lower() in {"1", "true", "yes", "on"}


def _open_without_proxy(request: Request, timeout: float):
    opener = build_opener(ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
