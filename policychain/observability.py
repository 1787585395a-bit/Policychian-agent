from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Iterator
from uuid import uuid4


DEFAULT_RUN_LOG_ROOT = Path("artifacts/run-logs")
_ACTIVE_RECORDER: ContextVar["RunRecorder | None"] = ContextVar("policychain_run_recorder", default=None)
_RUN_ID_PATTERN = re.compile(r"^run-[A-Za-z0-9_-]{8,80}$")
_SECRET_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|authorization|cookie|token|secret|password|mcp[_-]?key|credential)",
    re.IGNORECASE,
)
_LARGE_CONTENT_KEYS = {
    "prompt",
    "system_prompt",
    "user_prompt",
    "response",
    "raw_response",
    "policy_text",
    "policy_content",
    "content",
}
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+\-/=]+")
_KEY_VALUE_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|authorization|cookie|token|secret|password)\b\s*[:=]\s*([^\s,;]+)"
)


class RunRecorder:
    """Fail-open structured recorder for one PolicyChain workflow run."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        mode: str = "deterministic",
        log_root: str | Path | None = None,
        include_content: bool | None = None,
    ) -> None:
        self.run_id = run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:12]}"
        if not _RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError("run_id contains unsupported characters")
        self.mode = mode
        self.log_root = Path(log_root or os.getenv("POLICYCHAIN_RUN_LOG_DIR") or DEFAULT_RUN_LOG_ROOT)
        self.run_dir = self.log_root / self.run_id
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        if include_content is None:
            self.include_content = _truthy(os.getenv("POLICYCHAIN_LOG_FULL_LLM_IO")) or _truthy(
                os.getenv("POLICYCHAIN_LOG_INCLUDE_CONTENT")
            )
        else:
            self.include_content = include_content
        self.started_at = _now()
        self.finished_at = ""
        self.agent_status: dict[str, str] = {}
        self.fallback_used = False
        self.event_count = 0
        self.status = "running"
        self.logging_errors: list[str] = []
        self._token: Any = None
        self.record("run.start", stage="workflow", status="running", mode=mode)

    @contextmanager
    def activate(self) -> Iterator["RunRecorder"]:
        token = _ACTIVE_RECORDER.set(self)
        try:
            yield self
        finally:
            _ACTIVE_RECORDER.reset(token)

    def record(self, event_type: str, *, stage: str = "", status: str = "", **fields: Any) -> None:
        payload = {
            "time": _now(),
            "run_id": self.run_id,
            "event_type": event_type,
            "stage": stage,
            "status": status,
            **fields,
        }
        safe_payload = redact_payload(
            payload,
            include_content=self.include_content and event_type.startswith("llm.call"),
        )
        self.event_count += 1
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(safe_payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        except Exception as exc:
            self._remember_logging_error(exc)

    def set_agent_status(self, agent: str, status: str, **fields: Any) -> None:
        self.agent_status[agent] = status
        self.record("agent.status", stage=agent, status=status, agent=agent, **fields)

    def mark_fallback(self, agent: str, reason: str, fallback: str) -> None:
        self.fallback_used = True
        self.record(
            "fallback.used",
            stage=agent,
            status="fallback",
            agent=agent,
            reason=reason,
            fallback=fallback,
        )

    def finish(self, status: str, error: str = "", **fields: Any) -> dict[str, Any]:
        self.status = status
        self.finished_at = _now()
        self.record("run.finish", stage="workflow", status=status, error=error, **fields)
        summary = {
            "run_id": self.run_id,
            "mode": self.mode,
            "status": status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "agent_status": dict(self.agent_status),
            "fallback_used": self.fallback_used,
            "event_count": self.event_count,
            "logging_errors": list(self.logging_errors),
        }
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.summary_path.write_text(
                json.dumps(redact_payload({**summary, **fields}, include_content=False), ensure_ascii=False, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            self._remember_logging_error(exc)
        return summary

    def _remember_logging_error(self, exc: Exception) -> None:
        message = f"{exc.__class__.__name__}: {str(exc)[:200]}"
        if message not in self.logging_errors:
            self.logging_errors.append(message)


def current_run_recorder() -> RunRecorder | None:
    return _ACTIVE_RECORDER.get()


def record_event(event_type: str, *, stage: str = "", status: str = "", **fields: Any) -> None:
    recorder = current_run_recorder()
    if recorder is None:
        return
    try:
        recorder.record(event_type, stage=stage, status=status, **fields)
    except Exception:
        return


def load_run_artifact(run_id: str, log_root: str | Path | None = None) -> dict[str, Any]:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("invalid run_id")
    run_dir = Path(log_root or os.getenv("POLICYCHAIN_RUN_LOG_DIR") or DEFAULT_RUN_LOG_ROOT) / run_id
    summary_path = run_dir / "summary.json"
    events_path = run_dir / "events.jsonl"
    if not summary_path.is_file() and not events_path.is_file():
        raise FileNotFoundError(f"run artifact not found: {run_id}")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {"run_id": run_id, "status": "incomplete"}
    events: list[dict[str, Any]] = []
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                events.append(item)
    return {"summary": summary, "events": events}


def redact_payload(value: Any, *, include_content: bool = False, key: str = "") -> Any:
    if _SECRET_KEY_PATTERN.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(item_key): redact_payload(item_value, include_content=include_content, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [redact_payload(item, include_content=include_content, key=key) for item in value]
    if isinstance(value, str):
        redacted = _KEY_VALUE_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", value)
        redacted = _BEARER_PATTERN.sub("Bearer [REDACTED]", redacted)
        if not include_content and key.lower() in _LARGE_CONTENT_KEYS:
            return {
                "omitted": True,
                "char_count": len(redacted),
                "sha256": sha256(redacted.encode("utf-8")).hexdigest(),
            }
        return redacted
    return value


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
