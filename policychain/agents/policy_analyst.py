from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import re
from typing import Any, Iterable

from policychain.schemas.agent_outputs import (
    EvidenceItem,
    PolicyAnalysisOutput,
    StrengthAssessment,
)
from policychain.mcp import MCPToolInvoker, consume_mcp_invoker_errors, is_unavailable_invoker
from policychain.safety import assert_no_investment_advice
from policychain.source_policy import (
    SourcePolicyError,
    build_source_policy_from_local_policy,
    build_source_policy_from_user_input,
    is_url_input,
)
from policychain.state import PolicyResearchState
from policychain.storage.sqlite_store import SQLitePolicyStore
from policychain.tools import collect_policy_web_evidence, read_policy_content, search_policy
from policychain.tools.mcp_tools import mcp_unavailable_uncertainty


GOAL_MARKERS = ("为了", "促进", "规范", "推动", "鼓励", "支持", "提升", "发展")
MEASURE_MARKERS = ("应当", "不得", "鼓励", "支持", "要求", "建立", "采取", "依法", "备案", "监督")
ENTITY_MARKERS = (
    "生成式人工智能服务提供者",
    "提供者",
    "服务提供者",
    "用户",
    "国家",
    "主管部门",
    "行业组织",
    "未成年人",
)


class PolicyAnalysisError(RuntimeError):
    """Raised when Policy Analyst cannot produce a structured result."""


ProgressCallback = Callable[[int, str, str], None]


def run_policy_analyst(
    state: PolicyResearchState,
    store: SQLitePolicyStore,
    top_k: int = 5,
    mcp_invoker: MCPToolInvoker | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PolicyAnalysisOutput:
    """Run a deterministic Policy Analyst pass and write the result into shared state."""

    if not state.user_query.strip():
        raise PolicyAnalysisError("Policy Analyst requires a non-empty user query")

    source_policy = _ensure_source_policy(state, store, top_k=top_k, progress_callback=progress_callback)
    if not source_policy:
        output = _empty_analysis(state.user_query)
        state.policy_analysis = output.to_dict()
        state.uncertainties.extend(output.uncertainties)
        return output

    similar_results = _similar_policy_matches(store, source_policy, top_k=top_k)
    state.similar_policy_matches = similar_results
    if not similar_results:
        state.uncertainties = _unique([*state.uncertainties, "未在本地知识库中找到相似政策。"])
    _emit_progress(state, progress_callback, 15, "检索相似政策", f"找到 {len(similar_results)} 条相似政策证据")

    web_query = str(source_policy.get("title") or state.user_query)
    web_evidence = collect_policy_web_evidence(web_query, invoker=mcp_invoker)
    state.policy_web_evidence = web_evidence
    state.external_evidence = _merge_external_evidence(state.external_evidence, web_evidence)
    if is_unavailable_invoker(mcp_invoker):
        state.uncertainties = _unique([*state.uncertainties, mcp_unavailable_uncertainty("Open-WebSearch")])
    state.uncertainties = _unique([*state.uncertainties, *consume_mcp_invoker_errors(mcp_invoker)])

    output = analyze_policy_content(
        user_query=state.user_query,
        metadata=source_policy.get("metadata") or {},
        chunks=source_policy.get("chunks") or [],
    )

    policy_id = str((source_policy.get("metadata") or {}).get("policy_id") or source_policy.get("policy_id") or "")
    state.policy_ids = sorted({policy_id, *state.policy_ids}) if policy_id else state.policy_ids
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
) -> dict[str, Any] | None:
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
            raise PolicyAnalysisError(str(exc)) from exc
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
    source_policy: dict[str, Any],
) -> None:
    text_len = len(str(source_policy.get("text") or ""))
    title = str(source_policy.get("title") or source_policy.get("policy_id") or "")
    _emit_progress(state, callback, 12, "正文质量校验", f"已读取 {text_len} 字政策正文：{title}")


def _similar_policy_matches(
    store: SQLitePolicyStore,
    source_policy: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
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


def _similar_policy_query(source_policy: dict[str, Any], core_terms: list[str]) -> str:
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
    core = _compact(core)
    return [core] if len(core) >= 4 else []


def _matches_core_terms(match: dict[str, Any], core_terms: list[str]) -> bool:
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
    if callback:
        callback(progress, stage, message)


def _merge_external_evidence(existing: list[dict[str, Any]], new_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    merged: list[dict[str, Any]] = []
    for item in [*existing, *new_items]:
        key = (
            str(item.get("server_name") or ""),
            str(item.get("tool_name") or ""),
            str(item.get("source_url") or item.get("title") or item.get("query") or ""),
        )
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def analyze_policy_content(
    user_query: str,
    metadata: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> PolicyAnalysisOutput:
    if not metadata:
        raise PolicyAnalysisError("Policy Analyst requires policy metadata")
    if not chunks:
        raise PolicyAnalysisError("Policy Analyst requires policy chunks")

    policy_id = str(metadata.get("policy_id") or "")
    source_url = metadata.get("source_url")
    identity = _policy_identity(metadata)
    sentences = _sentences_from_chunks(chunks)
    goals = _extract_sentences(sentences, GOAL_MARKERS, limit=6)
    measures = _extract_sentences(sentences, MEASURE_MARKERS, limit=8)
    entities = _extract_entities(sentences)
    evidence = _build_evidence(policy_id=policy_id, source_url=source_url, chunks=chunks, user_query=user_query)
    strength = _assess_strength(sentences)
    uncertainties = _analysis_uncertainties(goals=goals, measures=measures)

    output = PolicyAnalysisOutput(
        policy_identity=identity,
        policy_goals=goals,
        target_entities=entities,
        policy_measures=measures,
        historical_changes=[],
        strength_assessment=strength,
        evidence=evidence,
        uncertainties=uncertainties,
    )
    assert_no_investment_advice(output.to_dict(), context="Policy analysis output")
    return output


def _empty_analysis(user_query: str) -> PolicyAnalysisOutput:
    return PolicyAnalysisOutput(
        policy_identity={"query": user_query, "status": "no_policy_found"},
        strength_assessment=StrengthAssessment(
            level="unknown",
            uncertainties=["未检索到可用于政策分析的本地政策证据。"],
        ),
        uncertainties=["未检索到可用于政策分析的本地政策证据。"],
    )


def _policy_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy_id": metadata.get("policy_id"),
        "title": metadata.get("title"),
        "document_number": metadata.get("document_number"),
        "publish_date": metadata.get("publish_date"),
        "issuing_agencies": metadata.get("issuing_agencies") or [],
        "policy_level": metadata.get("policy_level"),
        "policy_type": metadata.get("policy_type"),
        "policy_status": metadata.get("policy_status"),
        "source_url": metadata.get("source_url"),
    }


def _sentences_from_chunks(chunks: list[dict[str, Any]]) -> list[str]:
    sentences: list[str] = []
    for chunk in chunks:
        content = str(chunk.get("content") or "")
        sentences.extend(_split_sentences(content))
    return _unique(sentences)


def _split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[。；;.!?？])\s*|\n+", text)
    return [_compact(piece) for piece in pieces if _compact(piece)]


def _extract_sentences(sentences: Iterable[str], markers: tuple[str, ...], limit: int) -> list[str]:
    matches: list[str] = []
    for sentence in sentences:
        if any(marker in sentence for marker in markers):
            matches.append(_clip(sentence))
        if len(_unique(matches)) >= limit:
            break
    return _unique(matches)[:limit]


def _extract_entities(sentences: Iterable[str]) -> list[str]:
    text = "\n".join(sentences)
    entities = [marker for marker in ENTITY_MARKERS if marker in text]
    if "提供者" in entities and "生成式人工智能服务提供者" in entities:
        entities.remove("提供者")
    return _unique(entities)


def _build_evidence(
    policy_id: str,
    source_url: str | None,
    chunks: list[dict[str, Any]],
    user_query: str,
    limit: int = 5,
) -> list[EvidenceItem]:
    query_terms = _query_terms(user_query)
    evidence: list[EvidenceItem] = []
    for chunk in chunks:
        content = _compact(str(chunk.get("content") or ""))
        if not content:
            continue
        if query_terms and not any(term in content for term in query_terms):
            if len(evidence) >= 1:
                continue
        evidence.append(
            EvidenceItem(
                policy_id=policy_id,
                chunk_id=chunk.get("chunk_id"),
                source_url=source_url,
                text=_clip(content, max_chars=220),
                note=chunk.get("section_title"),
            )
        )
        if len(evidence) >= limit:
            break
    return evidence


def _assess_strength(sentences: list[str]) -> StrengthAssessment:
    text = "\n".join(sentences)
    hard_markers = sum(text.count(marker) for marker in ("应当", "不得", "依法", "监督", "处罚", "备案"))
    soft_markers = sum(text.count(marker) for marker in ("鼓励", "支持", "促进", "推动"))

    if hard_markers >= 6:
        level = "high"
        reasons = ["文本包含多处义务性、禁止性或监管性表述。"]
    elif hard_markers >= 2:
        level = "medium"
        reasons = ["文本包含明确的责任要求或治理措施，但仍需结合配套执行细则判断力度。"]
    elif soft_markers > 0:
        level = "low"
        reasons = ["文本更多体现鼓励、支持和方向性引导。"]
    else:
        level = "unknown"
        reasons = []

    uncertainties = ["当前为基于单份政策文本的初步判断，未纳入历史政策对比和后续配套文件。"]
    return StrengthAssessment(level=level, reasons=reasons, uncertainties=uncertainties)


def _analysis_uncertainties(goals: list[str], measures: list[str]) -> list[str]:
    uncertainties = ["历史政策对比尚未接入，historical_changes 暂为空。"]
    if not goals:
        uncertainties.append("未从已读取片段中稳定提取政策目标。")
    if not measures:
        uncertainties.append("未从已读取片段中稳定提取政策措施。")
    return uncertainties


def _query_terms(query: str) -> list[str]:
    return [term for term in re.split(r"\s+", query.strip()) if term]


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        compacted = _compact(value)
        if compacted and compacted not in seen:
            seen.add(compacted)
            output.append(compacted)
    return output


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clip(text: str, max_chars: int = 180) -> str:
    compacted = _compact(text)
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 1].rstrip() + "…"
