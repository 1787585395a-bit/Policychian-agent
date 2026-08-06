from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import (  # noqa: E402
    DEFAULT_POLICY_INPUT,
    _create_job,
    _health_payload,
    _job_view,
    _load_run_log,
    _payload_bool,
    render_example_report_page,
    render_page,
    run_query,
)


StartResponse = Callable[[str, list[tuple[str, str]]], Any]


def app(environ: dict[str, Any], start_response: StartResponse) -> Iterable[bytes]:
    """Vercel Python Runtime WSGI entrypoint."""

    method = str(environ.get("REQUEST_METHOD") or "GET").upper()
    path = str(environ.get("PATH_INFO") or "/")
    normalized_path = path.strip("/")
    if method == "GET":
        return _handle_get(environ, normalized_path, start_response)
    if method == "POST":
        return _handle_post(environ, normalized_path, start_response)
    return _response(start_response, b"Method Not Allowed", status="405 Method Not Allowed", content_type="text/plain")


def _handle_get(environ: dict[str, Any], path: str, start_response: StartResponse) -> Iterable[bytes]:
    if path == "healthz":
        return _json_response(start_response, _health_payload())
    if path == "api/research-status":
        params = parse_qs(str(environ.get("QUERY_STRING") or ""))
        job_id = (params.get("job_id") or [""])[0]
        return _json_response(start_response, _job_view(job_id))
    if path == "api/run-log":
        params = parse_qs(str(environ.get("QUERY_STRING") or ""))
        job_id = (params.get("job_id") or [""])[0]
        payload, status_code, filename = _load_run_log(job_id=job_id)
        return _attachment_json_response(start_response, payload, status_code=status_code, filename=filename)
    if path == "api/run-logs" or path.startswith("api/run-logs/"):
        params = parse_qs(str(environ.get("QUERY_STRING") or ""))
        run_id = path.removeprefix("api/run-logs/") if path.startswith("api/run-logs/") else (params.get("run_id") or [""])[0]
        payload, status_code, filename = _load_run_log(run_id=run_id)
        return _attachment_json_response(start_response, payload, status_code=status_code, filename=filename)
    if path == "example-report":
        return _response(start_response, render_example_report_page(), content_type="text/html; charset=utf-8")
    return _response(start_response, render_page(), content_type="text/html; charset=utf-8")


def _handle_post(environ: dict[str, Any], path: str, start_response: StartResponse) -> Iterable[bytes]:
    payload = _request_payload(environ)
    query = str(payload.get("query") or DEFAULT_POLICY_INPUT)
    use_llm = _payload_bool(payload, "use_llm", default=True)
    use_mcp = _payload_bool(payload, "use_mcp", default=True)

    if path == "api/research":
        job_id = _create_job(query=query, use_llm=use_llm, use_mcp=use_mcp)
        return _json_response(start_response, {"job_id": job_id, "status": "pending"}, status="202 Accepted")

    try:
        result = run_query(query, use_llm=use_llm, use_mcp=use_mcp)
        html = render_page(
            query=result["query"],
            report=result["report"],
            elapsed_seconds=result["elapsed_seconds"],
            use_llm=result["use_llm"],
            use_mcp=result["use_mcp"],
        )
    except Exception as exc:
        html = render_page(query=query, error=str(exc), use_llm=use_llm, use_mcp=use_mcp)
    return _response(start_response, html, content_type="text/html; charset=utf-8")


def _request_payload(environ: dict[str, Any]) -> dict[str, Any]:
    body = _read_body(environ)
    content_type = str(environ.get("CONTENT_TYPE") or "")
    if "application/json" in content_type:
        parsed = json.loads(body.decode("utf-8") or "{}")
        return parsed if isinstance(parsed, dict) else {}
    form = parse_qs(body.decode("utf-8", errors="replace"))
    return {
        "query": (form.get("query") or [DEFAULT_POLICY_INPUT])[0],
        "use_llm": (form.get("use_llm") or ["1"])[0] != "0",
        "use_mcp": (form.get("use_mcp") or ["1"])[0] != "0",
    }


def _read_body(environ: dict[str, Any]) -> bytes:
    try:
        length = int(environ.get("CONTENT_LENGTH") or "0")
    except ValueError:
        length = 0
    if length <= 0:
        return b""
    return environ["wsgi.input"].read(length)


def _json_response(start_response: StartResponse, payload: dict[str, Any], status: str = "200 OK") -> Iterable[bytes]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return _response(start_response, body, status=status, content_type="application/json; charset=utf-8")


def _attachment_json_response(
    start_response: StartResponse,
    payload: dict[str, Any],
    *,
    status_code: int,
    filename: str,
) -> Iterable[bytes]:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return _response(
        start_response,
        body,
        status=_status_line(status_code),
        content_type="application/json; charset=utf-8",
        extra_headers=[("Content-Disposition", f'attachment; filename="{filename}"')],
    )


def _status_line(status_code: int) -> str:
    return {
        200: "200 OK",
        400: "400 Bad Request",
        404: "404 Not Found",
    }.get(status_code, f"{status_code} Error")


def _response(
    start_response: StartResponse,
    body: bytes,
    status: str = "200 OK",
    content_type: str = "text/html; charset=utf-8",
    extra_headers: list[tuple[str, str]] | None = None,
) -> Iterable[bytes]:
    start_response(
        status,
        [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
            *(extra_headers or []),
        ],
    )
    return [body]
