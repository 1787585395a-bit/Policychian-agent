from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
import re
from typing import Iterable

from policychain.llm import LLMClient, create_llm_client, observed_llm_generate
from policychain.mcp import MCPToolInvoker, consume_mcp_invoker_errors, is_unavailable_invoker
from policychain.observability import record_event
from policychain.prompts import render_prompt
from policychain.schemas.agent_outputs import PolicyAnalysisOutput, StrengthAssessment
from policychain.source_policy import (
    SourcePolicyError,
    build_source_policy_from_local_policy,
    build_source_policy_from_user_input,
    is_url_input,
)
from policychain.state import PolicyResearchState
from policychain.storage.sqlite_store import SQLitePolicyStore
from policychain.structured_output import parse_structured_output
from policychain.tools import collect_policy_web_evidence, read_policy_content, run_policy_react_search, search_policy
from policychain.tools.mcp_tools import mcp_unavailable_uncertainty


class LLMPolicyAnalysisError(RuntimeError):
    """Raised when LLM Policy Analyst output cannot be trusted."""


ProgressCallback = Callable[[int, str, str], None]


def run_llm_policy_analyst(
    state: PolicyResearchState,
    store: SQLitePolicyStore,
    llm_client: LLMClient | None = None,
    top_k: int = 5,
    mcp_invoker: MCPToolInvoker | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PolicyAnalysisOutput:
    """Run the optional LLM Policy Analyst over retrieved policy evidence."""

    if not state.user_query.strip():
        raise LLMPolicyAnalysisError("LLM Policy Analyst requires a non-empty user query")

    source_policy = _ensure_source_policy(state, store, top_k=top_k, progress_callback=progress_callback)
    if not source_policy:
        output = _empty_analysis(state.user_query)
        state.policy_analysis = output.to_dict()
        state.uncertainties = _unique([*state.uncertainties, *output.uncertainties])
        return output

    similar_results = _similar_policy_matches(store, source_policy, top_k=top_k)
    state.similar_policy_matches = similar_results
    if not similar_results:
        state.uncertainties = _unique([*state.uncertainties, "未在本地知识库中找到相似政策。"])
    _emit_progress(state, progress_callback, 15, "检索相似政策", f"找到 {len(similar_results)} 条相似政策证据")

    client = llm_client or create_llm_client()
    web_query = str(source_policy.get("title") or state.user_query)
    web_evidence = collect_policy_web_evidence(web_query, invoker=mcp_invoker)
    if not is_unavailable_invoker(mcp_invoker):
        _emit_progress(state, progress_callback, 18, "ReAct 检索", "正在观察 Web Search 结果并补查相似政策证据")
        react_run = run_policy_react_search(web_query, invoker=mcp_invoker, llm_client=client)
        state.react_traces.extend(_tag_react_traces("policy", react_run.traces))
        web_evidence = _merge_external_evidence(web_evidence, react_run.evidence)
    state.policy_web_evidence = web_evidence
    state.external_evidence = _merge_external_evidence(state.external_evidence, web_evidence)
    if is_unavailable_invoker(mcp_invoker):
        state.uncertainties = _unique([*state.uncertainties, mcp_unavailable_uncertainty("Open-WebSearch")])
    state.uncertainties = _unique([*state.uncertainties, *consume_mcp_invoker_errors(mcp_invoker)])

    prompt = render_prompt(
        "policy_analyst",
        user_query=state.user_query,
        metadata=_json_for_prompt(source_policy.get("metadata") or {}),
        chunks=_json_for_prompt(source_policy.get("chunks") or []),
        source_policy=_json_for_prompt(source_policy),
        similar_policy_matches=_json_for_prompt(similar_results),
        web_evidence=_json_for_prompt(web_evidence),
    )
    raw_output = observed_llm_generate(client, prompt["system"], prompt["user"], agent="policy_analyst")
    output = parse_structured_output(raw_output, prompt["output_schema_name"])
    if not isinstance(output, PolicyAnalysisOutput):
        raise LLMPolicyAnalysisError("LLM Policy Analyst returned an unexpected output schema")
    policy_id = str((source_policy.get("metadata") or {}).get("policy_id") or source_policy.get("policy_id") or "")
    _assert_policy_identity_matches_retrieval(output, expected_policy_id=policy_id)
    _normalize_main_policy_evidence_ids(output, source_policy=source_policy, expected_policy_id=policy_id)
    _assert_policy_id_matches_retrieval(output, expected_policy_id=policy_id)

    state.policy_ids = sorted({policy_id, *state.policy_ids})
    state.policy_chunks = source_policy.get("chunks") or []
    state.policy_analysis = output.to_dict()
    state.evidence = [item.to_dict() for item in output.evidence]
    state.uncertainties = _unique([*state.uncertainties, *output.uncertainties])
    return output


def _ensure_source_policy(
    state: PolicyResearchState,
    store: SQLitePolicyStore,
    top_k: int,
    progress_callback: ProgressCallback | None,
) -> dict[str, object] | None:
    if state.source_policy:
        _emit_policy_quality_progress(state, progress_callback, state.source_policy)
        return state.source_policy
    source_is_url = is_url_input(state.user_query)
    if source_is_url:
        _emit_progress(state, progress_callback, 8, "URL 抓取", "正在抓取政策链接并提取正文")
    else:
        _emit_progress(state, progress_callback, 8, "URL 抓取", "输入为政策正文，跳过 URL 抓取")
    try:
        state.source_policy = build_source_policy_from_user_input(state.user_query)
        _emit_policy_quality_progress(state, progress_callback, state.source_policy)
        return state.source_policy
    except SourcePolicyError as exc:
        if source_is_url or len(state.user_query.strip()) >= 80:
            _emit_progress(state, progress_callback, 12, "正文质量校验", f"未通过正文质量校验：{exc}")
            raise LLMPolicyAnalysisError(str(exc)) from exc
        search_results = search_policy(store, state.user_query, top_k=top_k)
        if not search_results:
            return None
        policy_id = search_results[0]["policy_id"]
        chunk_ids = [result["chunk_id"] for result in search_results if result["policy_id"] == policy_id]
        content = read_policy_content(store, policy_id=policy_id, chunk_ids=chunk_ids, include_neighbors=True)
        state.source_policy = build_source_policy_from_local_policy(content, raw_input=state.user_query)
        _emit_policy_quality_progress(state, progress_callback, state.source_policy)
        return state.source_policy


def _emit_policy_quality_progress(
    state: PolicyResearchState,
    callback: ProgressCallback | None,
    source_policy: dict[str, object],
) -> None:
    text_len = len(str(source_policy.get("text") or ""))
    title = str(source_policy.get("title") or source_policy.get("policy_id") or "")
    _emit_progress(state, callback, 12, "正文质量校验", f"已读取 {text_len} 字政策正文：{title}")


def _similar_policy_matches(
    store: SQLitePolicyStore,
    source_policy: dict[str, object],
    top_k: int,
) -> list[dict[str, object]]:
    core_terms = _title_core_terms(str(source_policy.get("title") or ""))
    query = _similar_policy_query(source_policy, core_terms)
    source_policy_id = str((source_policy.get("metadata") or {}).get("policy_id") or source_policy.get("policy_id") or "")
    matches = search_policy(store, query, top_k=top_k)
    return [
        match
        for match in matches
        if str(match.get("policy_id") or "") != source_policy_id
        and float(match.get("score") or 0.0) >= 5.0
        and _matches_core_terms(match, core_terms)
    ]


def _similar_policy_query(source_policy: dict[str, object], core_terms: list[str]) -> str:
    text = f"{source_policy.get('title') or ''}\n{source_policy.get('text') or ''}"
    known_terms = [
        term
        for term in (
            "生成式人工智能",
            "人工智能",
            "算法",
            "模型",
            "训练数据",
            "个人信息",
            "数据安全",
            "网络信息",
            "量子算力",
            "低空经济",
        )
        if term in text
    ]
    return " ".join(_unique([*core_terms, *known_terms, str(source_policy.get("title") or "")]))


def _title_core_terms(title: str) -> list[str]:
    core = title
    for generic in ("管理暂行办法", "管理办法", "暂行办法", "实施细则", "若干措施", "通知", "意见", "规定", "办法", "服务", "基础设施", "管理"):
        core = core.replace(generic, "")
    core = " ".join(core.split()).strip()
    return [core] if len(core) >= 4 else []


def _matches_core_terms(match: dict[str, object], core_terms: list[str]) -> bool:
    if not core_terms:
        return True
    haystack = " ".join(str(match.get(key) or "") for key in ("title", "matched_text", "section_title"))
    return any(term in haystack for term in core_terms)


def _emit_progress(
    state: PolicyResearchState,
    callback: ProgressCallback | None,
    progress: int,
    stage: str,
    message: str,
) -> None:
    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "progress": progress,
        "stage": stage,
        "message": message,
    }
    state.progress_events.append(event)
    record_event("workflow.progress", stage=stage, status="ok", progress=progress, message=message)
    if callback:
        callback(progress, stage, message)


def _assert_policy_identity_matches_retrieval(output: PolicyAnalysisOutput, expected_policy_id: str) -> None:
    actual_policy_id = str(output.policy_identity.get("policy_id") or "")
    if actual_policy_id != expected_policy_id:
        raise LLMPolicyAnalysisError(
            f"LLM Policy Analyst policy_id mismatch: expected {expected_policy_id}, got {actual_policy_id or 'empty'}"
        )


def _normalize_main_policy_evidence_ids(
    output: PolicyAnalysisOutput,
    source_policy: dict[str, object],
    expected_policy_id: str,
) -> None:
    chunks = [item for item in source_policy.get("chunks") or [] if isinstance(item, dict)]
    main_chunks = [
        item
        for item in chunks
        if str(item.get("policy_id") or expected_policy_id).strip() == expected_policy_id
    ]
    main_chunk_ids = {
        str(item.get("chunk_id") or "").strip()
        for item in main_chunks
        if str(item.get("chunk_id") or "").strip()
    }
    metadata = source_policy.get("metadata") if isinstance(source_policy.get("metadata"), dict) else {}
    main_source_urls = {
        value
        for value in (
            _normalized_source_url(source_policy.get("source_url")),
            _normalized_source_url(metadata.get("source_url")),
            *(_normalized_source_url(item.get("source_url")) for item in main_chunks),
        )
        if value
    }
    main_texts = [
        value
        for value in (
            _normalized_evidence_text(source_policy.get("text")),
            *(_normalized_evidence_text(item.get("content")) for item in main_chunks),
        )
        if value
    ]

    for evidence in output.evidence:
        if evidence.policy_id == expected_policy_id:
            continue
        if _evidence_belongs_to_main_policy(
            evidence.chunk_id,
            evidence.source_url,
            evidence.text,
            main_chunk_ids=main_chunk_ids,
            main_source_urls=main_source_urls,
            main_texts=main_texts,
        ):
            evidence.policy_id = expected_policy_id


def _evidence_belongs_to_main_policy(
    chunk_id: str | None,
    source_url: str | None,
    text: str,
    *,
    main_chunk_ids: set[str],
    main_source_urls: set[str],
    main_texts: list[str],
) -> bool:
    normalized_chunk_id = str(chunk_id or "").strip()
    normalized_source_url = _normalized_source_url(source_url)

    if normalized_chunk_id and normalized_chunk_id not in main_chunk_ids:
        return False
    if normalized_source_url and normalized_source_url not in main_source_urls:
        return False

    chunk_matches = bool(normalized_chunk_id and normalized_chunk_id in main_chunk_ids)
    source_url_matches = bool(normalized_source_url and normalized_source_url in main_source_urls)
    normalized_text = _normalized_evidence_text(text)
    text_matches = bool(
        len(normalized_text) >= 12
        and any(normalized_text in main_text for main_text in main_texts)
    )
    return chunk_matches or source_url_matches or text_matches


def _normalized_source_url(value: object) -> str:
    return str(value or "").strip()


def _normalized_evidence_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def _assert_policy_id_matches_retrieval(output: PolicyAnalysisOutput, expected_policy_id: str) -> None:
    _assert_policy_identity_matches_retrieval(output, expected_policy_id)
    mismatched_evidence = [
        item.policy_id or "empty"
        for item in output.evidence
        if item.policy_id != expected_policy_id
    ]
    if mismatched_evidence:
        raise LLMPolicyAnalysisError(
            f"LLM Policy Analyst evidence policy_id mismatch: {', '.join(sorted(set(mismatched_evidence)))}"
        )


def _empty_analysis(user_query: str) -> PolicyAnalysisOutput:
    return PolicyAnalysisOutput(
        policy_identity={"query": user_query, "status": "no_policy_found"},
        strength_assessment=StrengthAssessment(
            level="unknown",
            uncertainties=["未检索到可用于政策分析的本地政策证据。"],
        ),
        uncertainties=["未检索到可用于政策分析的本地政策证据。"],
    )


def _json_for_prompt(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _merge_external_evidence(existing: list[dict[str, object]], new_items: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    seen: set[tuple[object, object, object]] = set()
    for item in [*existing, *new_items]:
        key = (item.get("server_name"), item.get("tool_name"), item.get("source_url") or item.get("title") or item.get("query"))
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _tag_react_traces(stage: str, traces: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{"stage": stage, **trace} for trace in traces]


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output
