from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import uuid4

from policychain.agents.company_matcher import (
    audit_company_match_output,
    match_candidate_records_to_impacts,
    resolve_company_discovery_mode,
    resolve_company_match_limit,
)
from policychain.llm import LLMClient, create_llm_client, observed_llm_generate
from policychain.mcp import MCPToolInvoker, consume_mcp_invoker_errors, is_unavailable_invoker, mcp_server_is_unavailable
from policychain.observability import record_event
from policychain.prompts import render_prompt
from policychain.schemas.agent_outputs import CompanyDiscoveryOutput, CompanyMatchOutput, CompanySeedOutput
from policychain.state import PolicyResearchState
from policychain.structured_output import parse_structured_output
from policychain.tools import (
    collect_company_candidates,
    collect_company_discovery_web_evidence,
    collect_company_web_evidence,
    merge_react_company_candidates,
    run_company_react_search,
)
from policychain.tools.mcp_tools import (
    CNFINANCIAL_SERVER,
    candidate_retrieval_statuses,
    mcp_unavailable_uncertainty,
    resolve_company_seeds,
    resolve_web_first_company_seeds,
)


DEFAULT_COMPANY_PROMPT_MAX_CHARS = 16000
INTERNAL_VERIFIED_SHORTLIST_TARGET = 4


class LLMCompanyMatchError(RuntimeError):
    """Raised when LLM Company Matcher output cannot be trusted."""


def run_llm_company_matcher(
    state: PolicyResearchState,
    llm_client: LLMClient | None = None,
    top_k_per_industry: int = 3,
    mcp_invoker: MCPToolInvoker | None = None,
) -> CompanyMatchOutput:
    """Run the optional LLM Company Matcher over retrieved company candidates."""

    top_k_per_industry = resolve_company_match_limit(top_k_per_industry)

    if not state.industry_impacts:
        output = CompanyMatchOutput(uncertainties=["缺少行业影响分析，无法生成公司业务匹配清单。"])
        _write_state(state, output, candidates=[], coverage=[], audit_logs=[])
        return output

    if resolve_company_discovery_mode() == "web_first":
        return _run_web_first_company_matcher(
            state,
            llm_client=llm_client,
            top_k_per_industry=top_k_per_industry,
            mcp_invoker=mcp_invoker,
        )

    tool_logs: list[dict[str, Any]] = []
    company_records = _company_records_for_impacts(
        state.industry_impacts,
        top_k_per_industry=top_k_per_industry,
        mcp_invoker=mcp_invoker,
        tool_logs=tool_logs,
    )
    client = llm_client or create_llm_client()
    react_evidence: list[dict[str, Any]] = []
    react_candidate_audit: list[dict[str, Any]] = []
    react_uncertainties: list[str] = []
    if not mcp_server_is_unavailable(mcp_invoker, CNFINANCIAL_SERVER):
        for impact_index, impact in enumerate(state.industry_impacts, start=1):
            impact_id = str(impact.get("impact_id") or f"IMP-{impact_index:03d}")
            scoped_records = [
                record
                for record in company_records
                if impact_id in (record.get("impact_ids") or [])
            ]
            react_query = _company_react_query([impact], scoped_records)
            try:
                react_run = run_company_react_search(
                    react_query,
                    invoker=mcp_invoker,
                    llm_client=client,
                    impact=impact,
                    impact_id=impact_id,
                    tool_logs=tool_logs,
                )
            except Exception as exc:
                reason_code = _company_react_failure_reason(exc)
                error_summary = _clip_prompt(str(exc) or exc.__class__.__name__, 180)
                state.react_traces.append(
                    {
                        "stage": "company",
                        "impact_id": impact_id,
                        "action": "optional_react_failed",
                        "error": error_summary,
                        "reason_code": reason_code,
                    }
                )
                react_uncertainties.append(
                    f"{impact_id} 公司 ReAct 可选检索失败，已继续执行公司 seed、身份核验与逐路径审查。"
                )
                record_event(
                    "react.failure",
                    stage="company_matcher",
                    status="error",
                    impact_id=impact_id,
                    reason_code=reason_code,
                    error_type=exc.__class__.__name__,
                    error=error_summary,
                    source="company_react_optional",
                )
                continue
            state.react_traces.extend(_tag_react_traces("company", react_run.traces, impact_id=impact_id))
            react_evidence = _merge_external_evidence(react_evidence, react_run.evidence)
        company_records, react_candidate_audit = merge_react_company_candidates(
            company_records,
            state.industry_impacts,
            react_evidence,
            invoker=mcp_invoker,
            tool_logs=tool_logs,
        )
    state.uncertainties = _unique([*state.uncertainties, *react_uncertainties])
    if react_evidence:
        state.company_research = _merge_external_evidence(state.company_research, react_evidence)
        state.external_evidence = _merge_external_evidence(state.external_evidence, react_evidence)
    retrieval_by_impact = candidate_retrieval_statuses(tool_logs)
    state.react_candidate_audit = react_candidate_audit
    for item in react_candidate_audit:
        record_event(
            "react.candidate",
            stage="company_matcher",
            status=str(item.get("decision") or ""),
            **item,
        )

    if state.run_mode == "llm" and mcp_invoker is None:
        seed_records = []
        seed_uncertainties = [
            "CNFinancial 与官方 Web 身份核验通道未配置，未生成无法完成身份闭环的公司 seed。"
        ]
    else:
        seed_records, seed_uncertainties = _generate_company_seeds(
            client,
            state.industry_impacts,
            company_records,
            react_evidence,
        )
    cached_seed_records, unresolved_seed_records, cached_seed_audit = _reuse_existing_verified_identities(
        seed_records,
        company_records,
    )
    company_records = _merge_company_records(company_records, cached_seed_records)
    resolved_seed_records, resolved_seed_audit = resolve_company_seeds(
        unresolved_seed_records,
        state.industry_impacts,
        invoker=mcp_invoker,
        tool_logs=tool_logs,
    )
    company_records = _merge_company_records(company_records, resolved_seed_records)
    state.company_seeds = seed_records
    state.company_seed_audit = [*cached_seed_audit, *resolved_seed_audit]
    state.company_identity_audit = [*cached_seed_audit, *resolved_seed_audit]
    state.company_evidence_bundles = []
    state.uncertainties = _unique([*state.uncertainties, *seed_uncertainties])

    company_records, rank_coverage, rank_audit = _rank_verified_company_records(
        state.industry_impacts,
        company_records,
        retrieval_by_impact=retrieval_by_impact,
    )
    state.tool_call_logs = _merge_tool_logs(state.tool_call_logs, tool_logs)
    if not company_records:
        output = CompanyMatchOutput(
            uncertainties=_unique(
                [
                    *_retrieval_uncertainties(retrieval_by_impact),
                    *seed_uncertainties,
                    "未形成同时通过当前 A 股身份与逐路径业务审查的公司候选。",
                ]
            )
        )
        _matches, coverage, audit_logs = match_candidate_records_to_impacts(
            state.industry_impacts,
            [],
            top_k_per_industry=top_k_per_industry,
            retrieval_by_impact=retrieval_by_impact,
        )
        _write_state(
            state,
            output,
            candidates=[],
            coverage=coverage or rank_coverage,
            audit_logs=[*rank_audit, *audit_logs],
        )
        state.uncertainties = _unique([*state.uncertainties, *consume_mcp_invoker_errors(mcp_invoker)])
        return output

    company_web_evidence = collect_company_web_evidence(company_records, invoker=mcp_invoker, tool_logs=tool_logs)
    company_web_evidence = _merge_external_evidence(company_web_evidence, react_evidence)
    state.tool_call_logs = _merge_tool_logs(state.tool_call_logs, tool_logs)
    state.company_research = company_web_evidence
    state.external_evidence = _merge_external_evidence(
        state.external_evidence,
        company_web_evidence,
    )
    if is_unavailable_invoker(mcp_invoker):
        state.uncertainties = _unique(
            [
                *state.uncertainties,
                mcp_unavailable_uncertainty("CNFinancial"),
                mcp_unavailable_uncertainty("Open-WebSearch"),
            ]
        )
    state.uncertainties = _unique([*state.uncertainties, *consume_mcp_invoker_errors(mcp_invoker)])

    prompt = _render_company_prompt(
        state.industry_impacts,
        company_records,
        company_web_evidence,
    )
    raw_output = observed_llm_generate(client, prompt["system"], prompt["user"], agent="company_matcher")
    output = parse_structured_output(raw_output, prompt["output_schema_name"])
    if not isinstance(output, CompanyMatchOutput):
        raise LLMCompanyMatchError("LLM Company Matcher returned an unexpected output schema")
    _assert_company_names_from_candidates(output, company_records)
    audited_output = audit_company_match_output(
        output,
        state.industry_impacts,
        company_records,
        top_k_per_industry=top_k_per_industry,
        retrieval_by_impact=retrieval_by_impact,
    )
    _write_state(
        state,
        audited_output,
        candidates=company_records,
        coverage=getattr(audited_output, "_company_coverage", []),
        audit_logs=[*rank_audit, *getattr(audited_output, "_audit_logs", [])],
    )
    return audited_output


def _run_web_first_company_matcher(
    state: PolicyResearchState,
    *,
    llm_client: LLMClient | None,
    top_k_per_industry: int,
    mcp_invoker: MCPToolInvoker | None,
) -> CompanyMatchOutput:
    client = llm_client or create_llm_client()
    tool_logs: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    seed_audit: list[dict[str, Any]] = []
    discovery_audit: list[dict[str, Any]] = []
    discovery_evidence: list[dict[str, Any]] = []
    discovery_uncertainties: list[str] = []

    for impact_index, impact in enumerate(state.industry_impacts, start=1):
        impact_id = str(impact.get("impact_id") or f"IMP-{impact_index:03d}")
        discovery_call_id = f"llm-discovery-{uuid4().hex}"
        prompt = render_prompt(
            "company_discovery",
            industry_impact=_json_compact(_compact_impact_for_prompt(impact, impact_index, 360)),
        )
        try:
            raw_output = observed_llm_generate(
                client,
                prompt["system"],
                prompt["user"],
                agent="company_discovery",
            )
            parsed = parse_structured_output(raw_output, prompt["output_schema_name"])
            if not isinstance(parsed, CompanyDiscoveryOutput):
                raise LLMCompanyMatchError("Company discovery returned an unexpected output schema")
            if parsed.impact_id != impact_id:
                raise LLMCompanyMatchError("Company discovery impact_id does not match the requested path")
        except Exception as exc:
            reason_code = f"discovery_{exc.__class__.__name__.lower()}"
            entry = _company_stage_entry(
                impact_id=impact_id,
                seed_id="",
                tool_call_id=discovery_call_id,
                source="llm",
                reason_code="discovery_error",
                status="error",
                cache_hit=False,
                error=_clip_prompt(str(exc) or exc.__class__.__name__, 180),
            )
            discovery_audit.append(entry)
            _record_company_event("company.discovery", entry)
            discovery_uncertainties.append(
                f"{impact_id} 公司 Web-first discovery 生成或结构校验失败；未回退到旧 CNFinancial-first 候选召回。"
            )
            continue

        discovery_uncertainties.extend(parsed.uncertainties)
        entry = _company_stage_entry(
            impact_id=impact_id,
            seed_id="",
            tool_call_id=discovery_call_id,
            source="llm",
            reason_code="discovery_completed" if parsed.seeds or parsed.web_queries else "web_empty",
            status="ok" if parsed.seeds or parsed.web_queries else "empty",
            cache_hit=False,
        )
        discovery_audit.append(entry)
        _record_company_event("company.discovery", entry)

        web_evidence, web_audit = collect_company_discovery_web_evidence(
            impact_id,
            parsed.web_queries[:2],
            invoker=mcp_invoker,
            top_k=5,
            tool_logs=tool_logs,
        )
        discovery_evidence = _merge_external_evidence(discovery_evidence, web_evidence)
        for web_entry in web_audit:
            discovery_audit.append(web_entry)
            _record_company_event("company.discovery", web_entry)

        scoped_seeds = [
            _seed_record(seed.to_dict(), discovery_call_id=discovery_call_id, source="llm")
            for seed in parsed.seeds
        ]
        scoped_seeds.extend(
            _web_evidence_seed_records(
                web_evidence,
                impact_id=impact_id,
                discovery_call_id=discovery_call_id,
            )
        )
        scoped_seeds = _dedupe_seed_records(scoped_seeds)[:6]
        for seed in scoped_seeds:
            seeds.append(seed)
            seed_entry = _company_stage_entry(
                impact_id=impact_id,
                seed_id=str(seed.get("seed_id") or ""),
                tool_call_id=str(seed.get("tool_call_id") or discovery_call_id),
                source=str(seed.get("source") or "llm_web_unverified"),
                reason_code="unverified_seed",
                status="unverified",
                cache_hit=False,
                company_name=str(seed.get("proposed_name") or ""),
                stock_code=str(seed.get("proposed_stock_code") or ""),
            )
            seed_audit.append(seed_entry)
            _record_company_event("company.seed", seed_entry)
        if not scoped_seeds:
            empty_entry = _company_stage_entry(
                impact_id=impact_id,
                seed_id="",
                tool_call_id=discovery_call_id,
                source="llm_web",
                reason_code="web_empty" if web_audit else "seed_output_empty",
                status="empty",
                cache_hit=False,
            )
            seed_audit.append(empty_entry)
            _record_company_event("company.seed", empty_entry)

    resolved_records, identity_audit, evidence_bundles = resolve_web_first_company_seeds(
        seeds,
        state.industry_impacts,
        discovery_evidence,
        invoker=mcp_invoker,
        tool_logs=tool_logs,
    )
    state.company_seeds = list(seeds)
    state.company_seed_audit = [*seed_audit, *identity_audit]
    state.company_discovery_audit = discovery_audit
    state.company_identity_audit = identity_audit
    state.company_evidence_bundles = evidence_bundles
    state.company_research = discovery_evidence
    state.external_evidence = _merge_external_evidence(state.external_evidence, discovery_evidence)
    state.tool_call_logs = _merge_tool_logs(state.tool_call_logs, tool_logs)
    state.uncertainties = _unique([*state.uncertainties, *discovery_uncertainties])
    state.uncertainties = _unique([*state.uncertainties, *consume_mcp_invoker_errors(mcp_invoker)])

    retrieval_by_impact = _web_first_retrieval_statuses(
        state.industry_impacts,
        discovery_audit,
        identity_audit,
    )
    ranked_records, rank_coverage, rank_audit = _rank_verified_company_records(
        state.industry_impacts,
        resolved_records,
        retrieval_by_impact=retrieval_by_impact,
        semantic_prefilter=False,
    )
    whitelist_audit: list[dict[str, Any]] = []
    final_audit: list[dict[str, Any]] = []
    evaluation_uncertainties: list[str] = []
    if ranked_records:
        evaluation_bundles = _verified_company_evaluation_bundles(
            state.industry_impacts,
            ranked_records,
            evidence_bundles,
        )
        evaluation_prompt = _render_verified_company_evaluation_prompt(
            state.industry_impacts,
            evaluation_bundles,
        )
        evaluation_raw = observed_llm_generate(
            client,
            evaluation_prompt["system"],
            evaluation_prompt["user"],
            agent="company_matcher",
        )
        evaluation_output = parse_structured_output(
            evaluation_raw,
            evaluation_prompt["output_schema_name"],
        )
        if not isinstance(evaluation_output, CompanyMatchOutput):
            raise LLMCompanyMatchError("LLM Company Matcher returned an unexpected output schema")
        whitelisted_output, whitelist_audit = _whitelist_company_evaluation_output(
            evaluation_output,
            ranked_records,
        )
        audited_output = audit_company_match_output(
            whitelisted_output,
            state.industry_impacts,
            ranked_records,
            top_k_per_industry=top_k_per_industry,
            retrieval_by_impact=retrieval_by_impact,
            allow_weak_semantic_keep_low=True,
        )
        final_matches = list(audited_output.companies)
        final_coverage = list(getattr(audited_output, "_company_coverage", []))
        final_audit = list(getattr(audited_output, "_audit_logs", []))
        evaluation_uncertainties = list(audited_output.uncertainties)
    else:
        final_matches, final_coverage, final_audit = match_candidate_records_to_impacts(
            state.industry_impacts,
            [],
            top_k_per_industry=top_k_per_industry,
            retrieval_by_impact=retrieval_by_impact,
        )
    coverage = _finalize_web_first_coverage(
        final_coverage or rank_coverage,
        discovery_audit,
        identity_audit,
    )
    combined_audit = [*rank_audit, *whitelist_audit, *final_audit]
    coverage = _annotate_web_first_coverage(
        coverage,
        identity_audit=identity_audit,
        rank_audit=rank_audit,
        whitelist_audit=whitelist_audit,
        final_audit=final_audit,
    )
    uncertainties = _unique(
        [
            *discovery_uncertainties,
            *evaluation_uncertainties,
            "公司发现采用 Web-first；CNFinancial 仅按明确公司名称或代码进行精确交叉核验。",
            "公司部分仅表示业务相关性研究清单，不构成任何投资建议。",
        ]
    )
    if not final_matches:
        uncertainties.append("未形成同时通过当前 A 股身份与逐路径业务证据审查的公司匹配。")
    if any(str(item.get("reason_code") or "") == "web_fallback" for item in identity_audit):
        uncertainties.append("部分低置信公司仅由两处独立 Web 证据支持，CNFinancial 未完成交叉验证。")
    output = CompanyMatchOutput(companies=final_matches, uncertainties=_unique(uncertainties))
    _write_state(
        state,
        output,
        candidates=ranked_records,
        coverage=coverage,
        audit_logs=combined_audit,
    )
    return output


def _company_records_for_impacts(
    industry_impacts: list[dict[str, Any]],
    top_k_per_industry: int,
    mcp_invoker: MCPToolInvoker | None = None,
    tool_logs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return collect_company_candidates(
        industry_impacts,
        invoker=mcp_invoker,
        top_k_per_industry=top_k_per_industry,
        tool_logs=tool_logs,
    )


def _company_stage_entry(
    *,
    impact_id: str,
    seed_id: str,
    tool_call_id: str,
    source: str,
    reason_code: str,
    status: str,
    cache_hit: bool,
    company_name: str = "",
    stock_code: str = "",
    error: str = "",
) -> dict[str, Any]:
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "impact_id": impact_id,
        "seed_id": seed_id,
        "tool_call_id": tool_call_id,
        "source": source,
        "reason_code": reason_code,
        "status": status,
        "cache_hit": cache_hit,
        "company_name": company_name,
        "stock_code": stock_code,
        "error": error,
    }


def _record_company_event(event_type: str, entry: dict[str, Any]) -> None:
    payload = {key: value for key, value in entry.items() if key not in {"status", "stage"}}
    record_event(
        event_type,
        stage="company_matcher",
        status=str(entry.get("status") or ""),
        **payload,
    )


def _seed_record(
    payload: dict[str, Any],
    *,
    discovery_call_id: str,
    source: str,
) -> dict[str, Any]:
    return {
        **payload,
        "seed_id": f"seed-{uuid4().hex}",
        "source": f"{source}_unverified",
        "tool_call_id": discovery_call_id,
        "time": datetime.now(timezone.utc).isoformat(),
        "status": "unverified",
    }


def _web_evidence_seed_records(
    evidence: list[dict[str, Any]],
    *,
    impact_id: str,
    discovery_call_id: str,
) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    for item in evidence:
        raw = item.get("raw_payload") if isinstance(item.get("raw_payload"), dict) else {}
        name = str(
            raw.get("company_name")
            or raw.get("current_name")
            or raw.get("stock_name")
            or raw.get("sec_name")
            or raw.get("secName")
            or raw.get("证券简称")
            or raw.get("股票简称")
            or ""
        ).strip()
        if not name:
            continue
        raw_code = str(
            raw.get("stock_code")
            or raw.get("code")
            or raw.get("symbol")
            or raw.get("sec_code")
            or raw.get("secCode")
            or raw.get("证券代码")
            or raw.get("股票代码")
            or ""
        )
        code = _normalize_seed_code(raw_code)
        if raw_code and not code:
            continue
        summary = _clip_prompt(item.get("summary") or item.get("title"), 240)
        seeds.append(
            _seed_record(
                {
                    "impact_id": impact_id,
                    "proposed_name": name,
                    "historical_names": [],
                    "proposed_stock_code": code,
                    "seed_reason": summary or "Web discovery 返回明确公司身份字段，仍待核验。",
                    "origin_channels": ["web"],
                    "source_url": str(item.get("source_url") or ""),
                },
                discovery_call_id=str(item.get("tool_call_id") or discovery_call_id),
                source="web",
            )
        )
    return seeds


def _dedupe_seed_records(seeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for seed in seeds:
        code = _normalize_seed_code(str(seed.get("proposed_stock_code") or ""))
        name = _normalize_identity_name(str(seed.get("proposed_name") or ""))
        key = (str(seed.get("impact_id") or ""), code or name)
        if not key[1]:
            continue
        if key not in index:
            index[key] = dict(seed)
            output.append(index[key])
            continue
        target = index[key]
        target["origin_channels"] = _unique(
            [*(target.get("origin_channels") or []), *(seed.get("origin_channels") or [])]
        )
        target["historical_names"] = _unique(
            [*(target.get("historical_names") or []), *(seed.get("historical_names") or [])]
        )[:3]
        target["seed_reason"] = _clip_prompt(
            "；".join(_unique([str(target.get("seed_reason") or ""), str(seed.get("seed_reason") or "")])),
            360,
        )
        target["source"] = "llm_web_unverified"
    return output


def _web_first_retrieval_statuses(
    industry_impacts: list[dict[str, Any]],
    discovery_audit: list[dict[str, Any]],
    identity_audit: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    precedence = (
        "web_fallback",
        "identity_conflict",
        "web_fallback_exhausted",
        "cnfinancial_unavailable",
        "cnfinancial_error",
        "cnfinancial_empty",
        "business_rejected",
        "discovery_error",
        "web_empty",
    )
    for index, impact in enumerate(industry_impacts, start=1):
        impact_id = str(impact.get("impact_id") or f"IMP-{index:03d}")
        scoped_discovery = [item for item in discovery_audit if str(item.get("impact_id") or "") == impact_id]
        scoped_identity = [item for item in identity_audit if str(item.get("impact_id") or "") == impact_id]
        reasons = [str(item.get("reason_code") or "") for item in [*scoped_identity, *scoped_discovery]]
        status = next((reason for reason in precedence if reason in reasons), "web_empty")
        output[impact_id] = {
            "status": status,
            "error": " | ".join(
                _unique(str(item.get("error") or "") for item in scoped_discovery if item.get("error"))
            )[:300],
            "query_terms": [str(item.get("query") or "") for item in scoped_discovery if item.get("query")],
            "query_count": sum(1 for item in scoped_discovery if item.get("query")),
            "requested_query_terms": [str(item.get("query") or "") for item in scoped_discovery if item.get("query")],
            "skipped_queries": [],
            "skipped_query_count": 0,
            "channel_statuses": {
                "discovery": _unique(str(item.get("status") or "") for item in scoped_discovery),
                "identity": _unique(str(item.get("status") or "") for item in scoped_identity),
            },
            "partial_failure_count": sum(
                1 for item in [*scoped_discovery, *scoped_identity] if str(item.get("status") or "") in {"error", "unavailable"}
            ),
        }
    return output


def _finalize_web_first_coverage(
    coverage: list[dict[str, Any]],
    discovery_audit: list[dict[str, Any]],
    identity_audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reasons = {
        "discovery_error": "公司 discovery 或结构校验失败；Web-first 模式未回退到旧 CNFinancial-first 召回。",
        "web_empty": "Web discovery 未返回可规范化的明确公司身份线索。",
        "cnfinancial_empty": "CNFinancial 精确查询返回空，且严格 Web fallback 未形成可靠身份与路径业务证据。",
        "cnfinancial_error": "CNFinancial 精确核验发生技术错误，且严格 Web fallback 未形成可靠证据。",
        "cnfinancial_unavailable": "CNFinancial 精确核验不可用或熔断，且严格 Web fallback 未形成可靠证据。",
        "web_fallback_exhausted": "已启动严格 Web fallback，但独立来源、代码校验或路径业务证据不足。",
        "identity_conflict": "公司名称、代码或当前上市身份存在冲突，已永久否决该线索。",
        "business_rejected": "公司身份线索存在，但缺少与本路径特异产品或业务的可靠证据。",
        "llm_not_selected": "公司已进入 LLM 逐路径评估池，但本轮未被模型选择进入最终审查清单。",
        "web_fallback": "CNFinancial 未完成交叉验证；仅由两处独立 Web 证据形成低置信业务匹配。",
        "selected": "已完成公司身份、业务证据和逐路径确定性审查。",
    }
    identity_by_impact: dict[str, list[str]] = {}
    for item in identity_audit:
        identity_by_impact.setdefault(str(item.get("impact_id") or ""), []).append(str(item.get("reason_code") or ""))
    for item in coverage:
        impact_id = str(item.get("impact_id") or "")
        if int(item.get("passed_count") or 0) > 0:
            status = "web_fallback" if "web_fallback" in identity_by_impact.get(impact_id, []) else "selected"
        else:
            status = str(item.get("retrieval_status") or "web_empty")
            if status == "web_fallback" or any(
                reason in {
                    "identity_verified",
                    "identity_verified_search_profile",
                    "identity_verified_code_profile",
                    "alias_code_merged",
                }
                for reason in identity_by_impact.get(impact_id, [])
            ):
                status = "business_rejected"
        item["retrieval_status"] = status
        item["coverage_status"] = status
        item["no_match_reason"] = (
            "" if int(item.get("passed_count") or 0) > 0 else reasons.get(status, item.get("no_match_reason") or "")
        )
    return coverage


def _annotate_web_first_coverage(
    coverage: list[dict[str, Any]],
    *,
    identity_audit: list[dict[str, Any]],
    rank_audit: list[dict[str, Any]],
    whitelist_audit: list[dict[str, Any]],
    final_audit: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    hard_veto_reasons = {
        "identity_conflict",
        "non_current_a_share_code",
        "non_current_a_share_identity",
        "path_provenance_mismatch",
        "llm_company_not_whitelisted",
        "llm_identity_not_whitelisted",
        "llm_path_not_whitelisted",
        "explicit_business_contradiction",
    }
    for item in coverage:
        impact_id = str(item.get("impact_id") or "")
        scoped_identity = [entry for entry in identity_audit if str(entry.get("impact_id") or "") == impact_id]
        scoped_rank = [entry for entry in rank_audit if str(entry.get("impact_id") or "") == impact_id]
        scoped_whitelist = [entry for entry in whitelist_audit if str(entry.get("impact_id") or "") == impact_id]
        scoped_final = [entry for entry in final_audit if str(entry.get("impact_id") or "") == impact_id]
        fallback_entries = [
            entry
            for entry in scoped_identity
            if str(entry.get("fallback_status") or "") in {"verified", "exhausted"}
        ]
        item["identity_verified_count"] = sum(
            1 for entry in scoped_identity if str(entry.get("status") or "") == "verified"
        )
        item["fallback_required_count"] = len(fallback_entries)
        item["fallback_started_count"] = len(fallback_entries)
        item["fallback_verified_count"] = sum(
            1 for entry in fallback_entries if str(entry.get("fallback_status") or "") == "verified"
        )
        item["fallback_exhausted_count"] = sum(
            1 for entry in fallback_entries if str(entry.get("fallback_status") or "") == "exhausted"
        )
        item["llm_input_count"] = sum(
            1 for entry in scoped_rank if str(entry.get("reason_code") or "") == "llm_input"
        )
        item["llm_not_selected_count"] = sum(
            1 for entry in scoped_whitelist if str(entry.get("reason_code") or "") == "llm_not_selected"
        )
        item["weak_keep_count"] = sum(
            1
            for entry in scoped_final
            if str(entry.get("reason_code") or "")
            in {"weak_semantic_keep_low", "web_weak_semantic_keep_low"}
        )
        item["hard_veto_count"] = sum(
            1
            for entry in [*scoped_identity, *scoped_whitelist, *scoped_final]
            if str(entry.get("reason_code") or "") in hard_veto_reasons
        )
        item["cap_trimmed_count"] = sum(
            1 for entry in scoped_final if str(entry.get("reason_code") or "") == "cap_trimmed"
        )
        item["final_count"] = int(item.get("passed_count") or 0)
        if not item["final_count"] and item["llm_not_selected_count"]:
            item["retrieval_status"] = "llm_not_selected"
            item["coverage_status"] = "llm_not_selected"
            item["no_match_reason"] = "公司已进入 LLM 逐路径评估池，但本轮未被模型选择进入最终清单。"
        record_event(
            "company.final_count",
            stage="company_matcher",
            status="ok" if item["final_count"] else "empty",
            impact_id=impact_id,
            final_count=item["final_count"],
            cap_trimmed_count=item["cap_trimmed_count"],
            llm_not_selected_count=item["llm_not_selected_count"],
        )
    return coverage


def _verified_company_evaluation_bundles(
    industry_impacts: list[dict[str, Any]],
    ranked_records: list[dict[str, Any]],
    evidence_bundles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    impact_by_id = {
        str(impact.get("impact_id") or f"IMP-{index:03d}"): {
            **impact,
            "impact_id": str(impact.get("impact_id") or f"IMP-{index:03d}"),
        }
        for index, impact in enumerate(industry_impacts, start=1)
    }
    accepted_statuses = {
        "identity_verified",
        "identity_verified_search_profile",
        "identity_verified_code_profile",
        "alias_code_merged",
        "web_fallback",
    }
    accepted_bundles = [
        item
        for item in evidence_bundles
        if str(item.get("status") or "") in accepted_statuses
    ]
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for record in ranked_records:
        company_name = str(record.get("company_name") or "")
        stock_code = str(record.get("stock_code") or "")
        for impact_id in record.get("impact_ids") or []:
            impact_id = str(impact_id or "")
            matching = next(
                (
                    item
                    for item in accepted_bundles
                    if str(item.get("impact_id") or "") == impact_id
                    and str(item.get("stock_code") or "") == stock_code
                ),
                None,
            )
            if matching is None:
                raise LLMCompanyMatchError(
                    f"Verified company evaluation bundle missing for {company_name} ({stock_code}) {impact_id}"
                )
            key = (impact_id, company_name, stock_code)
            if key in seen:
                continue
            seen.add(key)
            provenance = next(
                (
                    item
                    for item in (record.get("provenance") or [])
                    if isinstance(item, dict) and str(item.get("impact_id") or "") == impact_id
                ),
                {},
            )
            path_specific_business = str(
                (record.get("business_evidence_by_impact") or {}).get(impact_id)
                or matching.get("path_specific_business")
                or record.get("business_evidence")
                or record.get("matched_business")
                or ""
            )
            negative_evidence = _unique(
                [
                    *(matching.get("negative_evidence") or []),
                    *(record.get("negative_evidence") or []),
                ]
            )
            if impact_id in (record.get("web_fallback_impacts") or []):
                negative_evidence = _unique(
                    [
                        *negative_evidence,
                        "CNFinancial 未完成交叉验证；当前仅有两处独立 Web 证据。",
                    ]
                )
            output.append(
                {
                    "impact_id": impact_id,
                    "identity": {
                        "company_name": company_name,
                        "stock_code": stock_code,
                        "verification": str(record.get("identity_verification") or "cnfinancial"),
                        "identity_verified": True,
                    },
                    "verification_status": str(matching.get("status") or ""),
                    "path": dict(matching.get("path") or impact_by_id.get(impact_id) or {}),
                    "path_specific_business": path_specific_business,
                    "negative_evidence": negative_evidence,
                    "data_date": str(record.get("data_date") or matching.get("data_date") or "unknown"),
                    "confidence_cap": float(record.get("confidence_cap") or 0.92),
                    "tool_status": dict(matching.get("tool_status") or {}),
                    "cnfinancial_info": list(matching.get("cnfinancial_info") or []),
                    "cnfinancial_profile": list(matching.get("cnfinancial_profile") or []),
                    "web_evidence": list(matching.get("web_evidence") or []),
                    "provenance": {
                        "seed_id": str(provenance.get("seed_id") or matching.get("seed_id") or ""),
                        "tool_call_id": str(provenance.get("tool_call_id") or ""),
                        "source_type": str(provenance.get("source_type") or ""),
                    },
                }
            )
    return output


def _render_verified_company_evaluation_prompt(
    industry_impacts: list[dict[str, Any]],
    evaluation_bundles: list[dict[str, Any]],
) -> dict[str, str]:
    if not evaluation_bundles:
        raise LLMCompanyMatchError("Company evaluation requires at least one verified evidence bundle")
    allowed_impact_ids = {str(item.get("impact_id") or "") for item in evaluation_bundles}
    scoped_impacts = [
        _compact_impact_for_prompt(impact, index, 240)
        for index, impact in enumerate(industry_impacts, start=1)
        if str(impact.get("impact_id") or f"IMP-{index:03d}") in allowed_impact_ids
    ]
    prompt: dict[str, str] = {}
    compact_bundles: list[dict[str, Any]] = []
    for text_limit, evidence_limit, minimal in (
        (240, 2, False),
        (140, 1, False),
        (80, 1, False),
        (80, 1, True),
        (40, 1, True),
    ):
        compact_bundles = [
            _compact_verified_evaluation_bundle(item, text_limit, evidence_limit, minimal=minimal)
            for item in evaluation_bundles
        ]
        prompt = render_prompt(
            "company_matcher",
            industry_impacts=_json_compact(scoped_impacts),
            company_records=_json_compact(compact_bundles),
            web_evidence="[]",
        )
        if _prompt_char_count(prompt) <= _company_prompt_max_chars():
            break
    total_chars = _prompt_char_count(prompt)
    if total_chars > _company_prompt_max_chars():
        raise LLMCompanyMatchError(
            f"Verified company evaluation prompt exceeds configured budget: "
            f"{total_chars}>{_company_prompt_max_chars()}"
        )
    record_event(
        "company.prompt_budget",
        stage="company_matcher",
        status="ok",
        phase="verified_business_evaluation",
        budget_chars=_company_prompt_max_chars(),
        total_chars=total_chars,
        impact_count=len(scoped_impacts),
        candidate_count=len(compact_bundles),
        candidate_truncated_count=0,
        web_evidence_count=sum(len(item.get("web_evidence") or []) for item in compact_bundles),
    )
    return prompt


def _compact_verified_evaluation_bundle(
    bundle: dict[str, Any],
    text_limit: int,
    evidence_limit: int,
    *,
    minimal: bool = False,
) -> dict[str, Any]:
    def compact_payload(items: list[dict[str, Any]]) -> list[str]:
        return [_clip_prompt(_json_compact(item), text_limit) for item in items[:evidence_limit]]

    compact_web: list[dict[str, Any]] = []
    for item in (bundle.get("web_evidence") or [])[:evidence_limit]:
        compact = _compact_web_evidence_for_prompt(item, text_limit)
        compact["tool_call_id"] = str(item.get("tool_call_id") or "")
        compact_web.append(compact)
    compact = {
        "impact_id": str(bundle.get("impact_id") or ""),
        "identity": dict(bundle.get("identity") or {}),
        "verification_status": str(bundle.get("verification_status") or ""),
        "path_specific_business": _clip_prompt(bundle.get("path_specific_business"), text_limit),
        "negative_evidence": [
            _clip_prompt(value, text_limit) for value in (bundle.get("negative_evidence") or [])[:3]
        ],
        "data_date": _clip_prompt(bundle.get("data_date") or "unknown", 24),
        "confidence_cap": bundle.get("confidence_cap"),
        "tool_status": dict(bundle.get("tool_status") or {}),
        "cnfinancial_info_evidence": compact_payload(list(bundle.get("cnfinancial_info") or [])),
        "cnfinancial_profile_evidence": compact_payload(list(bundle.get("cnfinancial_profile") or [])),
        "web_evidence": compact_web,
        "provenance": dict(bundle.get("provenance") or {}),
    }
    if minimal:
        compact["negative_evidence"] = compact["negative_evidence"][:1]
        compact.pop("cnfinancial_info_evidence", None)
        compact.pop("cnfinancial_profile_evidence", None)
        compact.pop("tool_status", None)
        if not compact_web:
            compact.pop("web_evidence", None)
        return compact
    compact["path"] = {
        "industry": _clip_prompt((bundle.get("path") or {}).get("industry"), text_limit),
        "chain_segment": _clip_prompt((bundle.get("path") or {}).get("chain_segment"), text_limit),
        "business_variables": [
            _clip_prompt(value, text_limit)
            for value in ((bundle.get("path") or {}).get("business_variables") or [])[:4]
        ],
    }
    return compact


def _whitelist_company_evaluation_output(
    output: CompanyMatchOutput,
    ranked_records: list[dict[str, Any]],
) -> tuple[CompanyMatchOutput, list[dict[str, Any]]]:
    accepted: list[Any] = []
    audit: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for match in output.companies:
        matching_by_name = [
            record
            for record in ranked_records
            if str(record.get("company_name") or "") == match.company_name
        ]
        reason_code = "llm_evaluation_whitelisted"
        record: dict[str, Any] = {}
        if not matching_by_name:
            reason_code = "llm_company_not_whitelisted"
        else:
            allowed_codes = _unique(str(item.get("stock_code") or "") for item in matching_by_name)
            if not match.stock_code and len(allowed_codes) == 1:
                match.stock_code = allowed_codes[0]
            matching_by_identity = [
                item for item in matching_by_name if str(item.get("stock_code") or "") == match.stock_code
            ]
            if not matching_by_identity:
                reason_code = "llm_identity_not_whitelisted"
            else:
                record = matching_by_identity[0]
                if match.impact_id not in {str(value) for value in record.get("impact_ids") or []}:
                    reason_code = "llm_path_not_whitelisted"
        key = (match.impact_id, match.company_name, match.stock_code)
        if reason_code == "llm_evaluation_whitelisted" and key in seen:
            reason_code = "llm_duplicate_match"
        decision = "accepted" if reason_code == "llm_evaluation_whitelisted" else "rejected"
        provenance = next(
            (
                item
                for item in (record.get("provenance") or [])
                if isinstance(item, dict) and str(item.get("impact_id") or "") == match.impact_id
            ),
            {},
        )
        entry = {
            "impact_id": match.impact_id,
            "company_name": match.company_name,
            "stock_code": match.stock_code,
            "decision": decision,
            "reason_code": reason_code,
            "seed_id": str(provenance.get("seed_id") or ""),
            "tool_call_id": str(provenance.get("tool_call_id") or ""),
            "source": "llm_company_evaluation",
            "cache_hit": False,
            "url": str(record.get("source_url") or ""),
            "date": str(record.get("data_date") or ""),
        }
        audit.append(entry)
        record_event(
            "company.audit",
            stage="company_matcher",
            status=decision,
            **{key: value for key, value in entry.items() if key != "decision"},
        )
        if decision == "accepted":
            seen.add(key)
            accepted.append(match)
    for record in ranked_records:
        company_name = str(record.get("company_name") or "")
        stock_code = str(record.get("stock_code") or "")
        for impact_id in _unique(str(value or "") for value in (record.get("impact_ids") or [])):
            key = (impact_id, company_name, stock_code)
            if key in seen:
                continue
            provenance = next(
                (
                    item
                    for item in (record.get("provenance") or [])
                    if isinstance(item, dict) and str(item.get("impact_id") or "") == impact_id
                ),
                {},
            )
            entry = {
                "impact_id": impact_id,
                "company_name": company_name,
                "stock_code": stock_code,
                "decision": "not_selected",
                "reason_code": "llm_not_selected",
                "seed_id": str(provenance.get("seed_id") or ""),
                "tool_call_id": str(provenance.get("tool_call_id") or ""),
                "source": "llm_company_evaluation",
                "cache_hit": False,
                "url": str(record.get("source_url") or ""),
                "date": str(record.get("data_date") or ""),
            }
            audit.append(entry)
            record_event(
                "company.audit",
                stage="company_matcher",
                status="not_selected",
                **{field: value for field, value in entry.items() if field != "decision"},
            )
    uncertainties = list(output.uncertainties)
    rejected_count = sum(1 for item in audit if item["decision"] == "rejected")
    if rejected_count:
        uncertainties.append(
            f"Company Matcher 有 {rejected_count} 条输出未通过公司名称、代码与 impact 白名单，已确定性剔除。"
        )
    return CompanyMatchOutput(companies=accepted, uncertainties=_unique(uncertainties)), audit


def _generate_company_seeds(
    client: LLMClient,
    industry_impacts: list[dict[str, Any]],
    company_records: list[dict[str, Any]],
    react_evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    seeds: list[dict[str, Any]] = []
    uncertainties: list[str] = []
    for impact_index, impact in enumerate(industry_impacts, start=1):
        impact_id = str(impact.get("impact_id") or f"IMP-{impact_index:03d}")
        existing_identities = _verified_identities_for_impact(company_records, impact_id)
        remaining_deficit = max(INTERNAL_VERIFIED_SHORTLIST_TARGET - len(existing_identities), 0)
        if remaining_deficit == 0:
            continue
        scoped_web = [
            item
            for item in react_evidence
            if str(item.get("impact_id") or "") == impact_id
            and str(item.get("server_name") or "") != CNFINANCIAL_SERVER
        ]
        prompt = render_prompt(
            "company_seed",
            industry_impact=_json_compact(_compact_impact_for_prompt(impact, impact_index, 320)),
            seed_context=_json_compact(
                {
                    "existing_verified_identities": existing_identities,
                    "remaining_deficit": remaining_deficit,
                    "verified_shortlist_target": INTERNAL_VERIFIED_SHORTLIST_TARGET,
                }
            ),
            web_seed_evidence=_json_compact(
                [_compact_web_evidence_for_prompt(item, 240) for item in scoped_web[:8]]
            ),
        )
        seed_call_id = f"llm-seed-{uuid4().hex}"
        try:
            raw_output = observed_llm_generate(
                client,
                prompt["system"],
                prompt["user"],
                agent="company_seed",
            )
            parsed = parse_structured_output(raw_output, prompt["output_schema_name"])
            if not isinstance(parsed, CompanySeedOutput):
                raise LLMCompanyMatchError("LLM company seed generator returned an unexpected output schema")
        except Exception as exc:
            reason_code = f"seed_generation_{exc.__class__.__name__.lower()}"
            record_event(
                "company.seed",
                stage="company_matcher",
                status="error",
                seed_id="",
                impact_id=impact_id,
                tool_call_id=seed_call_id,
                reason_code=reason_code,
                cache_hit=False,
                source="llm",
                url="",
                date=datetime.now(timezone.utc).isoformat(),
            )
            uncertainties.append(f"{impact_id} 公司 seed 生成或结构校验失败，未将任何未验证线索提升为候选。")
            continue

        uncertainties.extend(parsed.uncertainties)
        accepted_for_impact = 0
        for seed in parsed.seeds:
            payload = seed.to_dict()
            payload["seed_id"] = f"seed-{uuid4().hex}"
            payload["source"] = "llm_web_unverified"
            payload["tool_call_id"] = seed_call_id
            payload["time"] = datetime.now(timezone.utc).isoformat()
            payload["status"] = "unverified"
            if seed.impact_id != impact_id:
                record_event(
                    "company.seed",
                    stage="company_matcher",
                    status="rejected",
                    seed_id=payload["seed_id"],
                    impact_id=seed.impact_id,
                    tool_call_id=seed_call_id,
                    reason_code="seed_path_mismatch",
                    cache_hit=False,
                    source="llm",
                    url="",
                    date=payload["time"],
                )
                continue
            seeds.append(payload)
            accepted_for_impact += 1
            record_event(
                "company.seed",
                stage="company_matcher",
                status="unverified",
                seed_id=payload["seed_id"],
                impact_id=impact_id,
                tool_call_id=seed_call_id,
                reason_code="llm_web_seed_unverified",
                cache_hit=False,
                source=",".join(payload.get("origin_channels") or ["llm"]),
                url="",
                date=payload["time"],
            )
        if not accepted_for_impact:
            record_event(
                "company.seed",
                stage="company_matcher",
                status="empty",
                seed_id="",
                impact_id=impact_id,
                tool_call_id=seed_call_id,
                reason_code="seed_output_empty",
                cache_hit=False,
                source="llm",
                url="",
                date=datetime.now(timezone.utc).isoformat(),
            )
    return seeds, _unique(uncertainties)


def _verified_identities_for_impact(
    company_records: list[dict[str, Any]],
    impact_id: str,
) -> list[dict[str, str]]:
    identities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for record in company_records:
        if not _record_identity_is_verified(record) or not _record_is_for_impact(record, impact_id):
            continue
        identity = _record_identity(record)
        if identity in seen:
            continue
        seen.add(identity)
        identities.append(
            {
                "company_name": str(record.get("company_name") or ""),
                "stock_code": _normalized_record_code(record),
            }
        )
    return identities


def _reuse_existing_verified_identities(
    seeds: list[dict[str, Any]],
    company_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    verified_records = [record for record in company_records if _record_identity_is_verified(record)]
    by_code: dict[str, dict[str, Any]] = {}
    by_name: dict[str, list[dict[str, Any]]] = {}
    for record in verified_records:
        code = _normalized_record_code(record)
        name = _normalize_identity_name(str(record.get("company_name") or ""))
        if code:
            by_code.setdefault(code, record)
        if name:
            by_name.setdefault(name, []).append(record)

    cached_records: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for seed in seeds:
        proposed_code = _normalize_seed_code(str(seed.get("proposed_stock_code") or ""))
        proposed_name = str(seed.get("proposed_name") or "")
        name_key = _normalize_identity_name(proposed_name)
        record: dict[str, Any] | None = None
        reason_code = ""
        if proposed_code and proposed_code in by_code:
            candidate = by_code[proposed_code]
            if name_key != _normalize_identity_name(str(candidate.get("company_name") or "")):
                reason_code = "duplicate_existing_name_code_conflict"
            else:
                record = candidate
        elif not proposed_code and len(by_name.get(name_key, [])) == 1:
            record = by_name[name_key][0]
        elif not proposed_code and len(by_name.get(name_key, [])) > 1:
            reason_code = "duplicate_existing_name_ambiguous"

        if reason_code:
            entry = _cached_seed_audit_entry(
                seed,
                status="rejected",
                reason_code=reason_code,
                record=by_code.get(proposed_code) or {},
            )
            audit.append(entry)
            _record_cached_seed_event("company.identity", entry)
            continue
        if record is None:
            unresolved.append(seed)
            continue

        impact_id = str(seed.get("impact_id") or "")
        provenance = {
            "impact_id": impact_id,
            "seed_id": str(seed.get("seed_id") or ""),
            "seed_reason": str(seed.get("seed_reason") or ""),
            "origin_channels": list(seed.get("origin_channels") or []),
            "tool": "existing_verified_identity_cache",
            "tool_call_id": str(seed.get("tool_call_id") or ""),
            "source_type": "llm_seed_existing_identity",
            "source_url": str(record.get("source_url") or ""),
            "data_date": str(record.get("data_date") or ""),
        }
        cached_records.append(
            {
                **record,
                "identity_verified": True,
                "impact_ids": _unique([*(record.get("impact_ids") or []), impact_id]),
                "seed_reasons": _unique(
                    [*(record.get("seed_reasons") or []), str(seed.get("seed_reason") or "")]
                ),
                "provenance": [provenance],
            }
        )
        entry = _cached_seed_audit_entry(
            seed,
            status="verified",
            reason_code="duplicate_existing_verified_identity",
            record=record,
        )
        audit.append(entry)
        _record_cached_seed_event("company.identity", entry)
        _record_cached_seed_event(
            "company.enrichment",
            {**entry, "status": "cached", "reason_code": "existing_identity_enrichment_cache"},
        )
    return cached_records, unresolved, audit


def _cached_seed_audit_entry(
    seed: dict[str, Any],
    *,
    status: str,
    reason_code: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "seed_id": str(seed.get("seed_id") or ""),
        "impact_id": str(seed.get("impact_id") or ""),
        "proposed_name": str(seed.get("proposed_name") or ""),
        "proposed_stock_code": str(seed.get("proposed_stock_code") or ""),
        "company_name": str(record.get("company_name") or ""),
        "stock_code": _normalized_record_code(record),
        "source": "existing_verified_candidate_cache",
        "url": str(record.get("source_url") or ""),
        "date": str(record.get("data_date") or seed.get("time") or ""),
        "tool_call_id": str(seed.get("tool_call_id") or ""),
        "cache_hit": True,
        "status": status,
        "reason_code": reason_code,
    }


def _record_cached_seed_event(event_type: str, entry: dict[str, Any]) -> None:
    record_event(
        event_type,
        stage="company_matcher",
        status=str(entry.get("status") or ""),
        seed_id=str(entry.get("seed_id") or ""),
        impact_id=str(entry.get("impact_id") or ""),
        tool_call_id=str(entry.get("tool_call_id") or ""),
        reason_code=str(entry.get("reason_code") or ""),
        cache_hit=True,
        source=str(entry.get("source") or ""),
        url=str(entry.get("url") or ""),
        date=str(entry.get("date") or ""),
        company_name=str(entry.get("company_name") or ""),
        stock_code=str(entry.get("stock_code") or entry.get("proposed_stock_code") or ""),
    )


def _rank_verified_company_records(
    industry_impacts: list[dict[str, Any]],
    company_records: list[dict[str, Any]],
    *,
    retrieval_by_impact: dict[str, dict[str, Any]],
    semantic_prefilter: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    identity_verified = [record for record in company_records if _record_identity_is_verified(record)]
    if not semantic_prefilter:
        return _rank_verified_records_for_llm(
            industry_impacts,
            identity_verified,
            retrieval_by_impact=retrieval_by_impact,
        )
    matches, coverage, audit_logs = match_candidate_records_to_impacts(
        industry_impacts,
        identity_verified,
        top_k_per_industry=4,
        retrieval_by_impact=retrieval_by_impact,
    )
    accepted_ids: dict[tuple[str, str], list[str]] = {}
    rank_by_key: dict[tuple[str, str, str], int] = {}
    per_impact_rank: dict[str, int] = {}
    for match in matches:
        identity = _record_identity({"company_name": match.company_name, "stock_code": match.stock_code})
        accepted_ids.setdefault(identity, [])
        accepted_ids[identity] = _unique([*accepted_ids[identity], match.impact_id])
        per_impact_rank[match.impact_id] = per_impact_rank.get(match.impact_id, 0) + 1
        rank_by_key[(match.impact_id, match.company_name, match.stock_code)] = per_impact_rank[match.impact_id]

    ranked: list[dict[str, Any]] = []
    for record in identity_verified:
        allowed_impact_ids = accepted_ids.get(_record_identity(record), [])
        if not allowed_impact_ids:
            continue
        ranked.append(
            {
                **record,
                "impact_ids": allowed_impact_ids,
                "provenance": [
                    item
                    for item in (record.get("provenance") or [])
                    if not isinstance(item, dict) or str(item.get("impact_id") or "") in allowed_impact_ids
                ],
            }
        )

    for audit in audit_logs:
        key = (
            str(audit.get("impact_id") or ""),
            str(audit.get("company_name") or ""),
            str(audit.get("stock_code") or ""),
        )
        record = next(
            (
                item
                for item in identity_verified
                if _record_identity(item) == _record_identity(
                    {"company_name": key[1], "stock_code": key[2]}
                )
            ),
            {},
        )
        provenance = next(
            (
                item
                for item in (record.get("provenance") or [])
                if isinstance(item, dict) and str(item.get("impact_id") or "") == key[0]
            ),
            {},
        )
        included_rank = rank_by_key.get(key)
        decision = str(audit.get("decision") or "reject")
        status = "accepted" if included_rank is not None else decision
        reason_code = str(audit.get("reason_code") or "")
        if decision != "reject" and included_rank is None:
            reason_code = "shortlist_limit"
        common_event = {
            "stage": "company_matcher",
            "status": status,
            "seed_id": str(provenance.get("seed_id") or ""),
            "impact_id": key[0],
            "tool_call_id": str(provenance.get("tool_call_id") or ""),
            "reason_code": reason_code or decision,
            "cache_hit": False,
            "source": str(provenance.get("source_type") or "deterministic_company_audit"),
            "url": str(record.get("source_url") or ""),
            "date": str(record.get("data_date") or ""),
            "company_name": key[1],
            "stock_code": key[2],
            "confidence": float(audit.get("confidence") or 0.0),
        }
        record_event("company.audit", **common_event)
        record_event(
            "company.rank",
            **common_event,
            rank=included_rank,
        )
    return ranked, coverage, audit_logs


def _rank_verified_records_for_llm(
    industry_impacts: list[dict[str, Any]],
    identity_verified: list[dict[str, Any]],
    *,
    retrieval_by_impact: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    audit_logs: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []

    for impact_index, impact in enumerate(industry_impacts, start=1):
        impact_id = str(impact.get("impact_id") or f"IMP-{impact_index:03d}")
        scoped_records = [record for record in identity_verified if _record_is_for_impact(record, impact_id)]
        selected_count = 0
        seen_identities: set[tuple[str, str]] = set()
        rejected_count = 0
        for record in scoped_records:
            identity_key = _record_identity(record)
            if identity_key in seen_identities:
                continue
            seen_identities.add(identity_key)
            business_text = str(
                (record.get("business_evidence_by_impact") or {}).get(impact_id)
                or record.get("business_evidence")
                or record.get("matched_business")
                or ""
            ).strip()
            provenance = next(
                (
                    item
                    for item in (record.get("provenance") or [])
                    if isinstance(item, dict)
                    and str(item.get("impact_id") or "") == impact_id
                    and any(item.get(field) for field in ("tool", "tool_call_id", "source_type"))
                ),
                None,
            )
            if not business_text:
                decision = "reject"
                reason_code = "pre_llm_missing_business_evidence"
            elif provenance is None:
                decision = "reject"
                reason_code = "pre_llm_missing_provenance"
            else:
                decision = "accepted"
                reason_code = "llm_input"
                selected_count += 1
                target = selected_by_identity.setdefault(identity_key, {**record, "impact_ids": [], "provenance": []})
                target["impact_ids"] = _unique([*(target.get("impact_ids") or []), impact_id])
                target["provenance"] = [
                    *(target.get("provenance") or []),
                    *[
                        item
                        for item in (record.get("provenance") or [])
                        if not isinstance(item, dict) or str(item.get("impact_id") or "") == impact_id
                    ],
                ]
            if decision == "reject":
                rejected_count += 1
            entry = {
                "impact_id": impact_id,
                "company_name": str(record.get("company_name") or ""),
                "stock_code": str(record.get("stock_code") or ""),
                "decision": decision,
                "reason_code": reason_code,
                "confidence": 0.0,
                "shared_terms": [],
            }
            audit_logs.append(entry)
            provenance_item = provenance or {}
            event_payload = {
                "stage": "company_matcher",
                "status": decision,
                "seed_id": str(provenance_item.get("seed_id") or ""),
                "impact_id": impact_id,
                "tool_call_id": str(provenance_item.get("tool_call_id") or ""),
                "reason_code": reason_code,
                "cache_hit": False,
                "source": str(provenance_item.get("source_type") or "llm_input"),
                "url": str(record.get("source_url") or ""),
                "date": str(record.get("data_date") or ""),
                "company_name": str(record.get("company_name") or ""),
                "stock_code": str(record.get("stock_code") or ""),
                "confidence": 0.0,
            }
            record_event("company.audit", **event_payload)
            record_event(
                "company.rank",
                **event_payload,
                rank=selected_count if decision == "accepted" else None,
            )

        retrieval = dict(retrieval_by_impact.get(impact_id) or {})
        coverage.append(
            {
                "impact_id": impact_id,
                "industry": str(impact.get("industry") or ""),
                "policy_measure": str(impact.get("policy_measure") or ""),
                "implementation_action": str(impact.get("implementation_action") or ""),
                "chain_segment": str(impact.get("chain_segment") or ""),
                "business_variables": list(impact.get("business_variables") or []),
                "affected_company_types": list(impact.get("affected_company_types") or []),
                "candidate_count": len(scoped_records),
                "passed_count": selected_count,
                "rejected_count": rejected_count,
                "company_names": [
                    str(record.get("company_name") or "")
                    for record in selected_by_identity.values()
                    if impact_id in (record.get("impact_ids") or [])
                ],
                "retrieval_status": str(retrieval.get("status") or ("ok" if scoped_records else "empty")),
                "retrieval_error": str(retrieval.get("error") or ""),
                "retrieval_queries": list(retrieval.get("query_terms") or []),
                "retrieval_query_count": int(retrieval.get("query_count") or 0),
                "retrieval_skipped_queries": list(retrieval.get("skipped_queries") or []),
                "retrieval_skipped_query_count": int(retrieval.get("skipped_query_count") or 0),
                "retrieval_channel_statuses": dict(retrieval.get("channel_statuses") or {}),
                "no_match_reason": "" if selected_count else "缺少可回溯的公司业务资料或 provenance。",
            }
        )
    return list(selected_by_identity.values()), coverage, audit_logs


def _record_identity_is_verified(record: dict[str, Any]) -> bool:
    if record.get("identity_verified") is True:
        return True
    return str(record.get("candidate_source_tool") or "") == "get_industry_stocks"


def _record_is_for_impact(record: dict[str, Any], impact_id: str) -> bool:
    impact_ids = {
        str(value)
        for value in record.get("impact_ids") or []
        if value
    }
    impact_ids.update(
        str(item.get("impact_id") or "")
        for item in record.get("provenance") or []
        if isinstance(item, dict) and item.get("impact_id")
    )
    return not impact_ids or impact_id in impact_ids


def _record_identity(record: dict[str, Any]) -> tuple[str, str]:
    code = _normalized_record_code(record)
    if code:
        return "code", code
    return "name", _normalize_identity_name(str(record.get("company_name") or ""))


def _normalized_record_code(record: dict[str, Any]) -> str:
    return _normalize_seed_code(str(record.get("stock_code") or ""))


def _normalize_seed_code(value: str) -> str:
    code_match = re.search(r"(?<!\d)(\d{6})(?!\d)", value)
    return code_match.group(1) if code_match else ""


def _normalize_identity_name(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]", "", str(value or "")).lower()


def _merge_company_records(
    existing: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for record in [*existing, *new_records]:
        identity = _record_identity(record)
        if identity not in merged:
            merged[identity] = dict(record)
            order.append(identity)
            continue
        target = merged[identity]
        if record.get("identity_verified") is True:
            target["identity_verified"] = True
        for field in ("company_name", "stock_code", "matched_business", "business_evidence", "source_url", "data_date"):
            if not target.get(field) and record.get(field):
                target[field] = record[field]
        for field in ("impact_ids", "business_keywords", "seed_reasons"):
            target[field] = _unique([*(target.get(field) or []), *(record.get(field) or [])])
        target["provenance"] = [*(target.get("provenance") or []), *(record.get("provenance") or [])]
        target["mcp_evidence"] = _merge_external_evidence(
            list(target.get("mcp_evidence") or []),
            list(record.get("mcp_evidence") or []),
        )
    return [merged[identity] for identity in order]


def _assert_company_names_from_candidates(
    output: CompanyMatchOutput,
    company_records: list[dict[str, Any]],
) -> None:
    records_by_name: dict[str, list[dict[str, Any]]] = {}
    for record in company_records:
        records_by_name.setdefault(str(record.get("company_name") or ""), []).append(record)
    unknown_names = sorted(company.company_name for company in output.companies if company.company_name not in records_by_name)
    if unknown_names:
        raise LLMCompanyMatchError(
            f"LLM Company Matcher returned company outside candidate records: {', '.join(unknown_names)}"
        )
    for company in output.companies:
        allowed_codes = _unique(
            str(record.get("stock_code") or "")
            for record in records_by_name.get(company.company_name, [])
        )
        if company.stock_code:
            if allowed_codes and company.stock_code not in allowed_codes:
                raise LLMCompanyMatchError(
                    f"LLM Company Matcher returned inconsistent name/code identity: "
                    f"{company.company_name} ({company.stock_code})"
                )
            continue
        if len(allowed_codes) == 1:
            company.stock_code = allowed_codes[0]
            continue
        if len(allowed_codes) > 1:
            raise LLMCompanyMatchError(
                f"LLM Company Matcher omitted code for ambiguous company identity: {company.company_name}"
            )


def _write_state(
    state: PolicyResearchState,
    output: CompanyMatchOutput,
    candidates: list[dict[str, Any]],
    coverage: list[dict[str, Any]] | None = None,
    audit_logs: list[dict[str, Any]] | None = None,
) -> None:
    payload = output.to_dict()
    state.company_candidates = list(candidates)
    state.company_matches = payload["companies"]
    state.company_coverage = list(coverage or [])
    state.company_match_audit = list(audit_logs or [])
    state.uncertainties = _unique([*state.uncertainties, *payload["uncertainties"]])


def _render_company_prompt(
    industry_impacts: list[dict[str, Any]],
    company_records: list[dict[str, Any]],
    web_evidence: list[dict[str, Any]],
) -> dict[str, str]:
    max_chars = _company_prompt_max_chars()
    prompt: dict[str, str] = {}
    compact_candidates: list[dict[str, Any]] = []
    compact_web: list[dict[str, Any]] = []
    profiles = (
        (320, 2, 12),
        (160, 1, 6),
        (80, 1, 0),
        (40, 0, 0),
    )
    for text_limit, evidence_limit, web_limit in profiles:
        compact_impacts = [
            _compact_impact_for_prompt(impact, index, text_limit)
            for index, impact in enumerate(industry_impacts, start=1)
        ]
        compact_candidates = [
            _compact_candidate_for_prompt(record, text_limit, evidence_limit)
            for record in company_records
        ]
        compact_web = [
            _compact_web_evidence_for_prompt(item, text_limit)
            for item in web_evidence[:web_limit]
        ]
        prompt = render_prompt(
            "company_matcher",
            industry_impacts=_json_compact(compact_impacts),
            company_records=_json_compact(compact_candidates),
            web_evidence=_json_compact(compact_web),
        )
        if _prompt_char_count(prompt) <= max_chars:
            break

    original_candidate_count = len(compact_candidates)
    while compact_candidates and _prompt_char_count(prompt) > max_chars:
        compact_candidates.pop()
        prompt = render_prompt(
            "company_matcher",
            industry_impacts=_json_compact(compact_impacts),
            company_records=_json_compact(compact_candidates),
            web_evidence="[]",
        )
    total_chars = _prompt_char_count(prompt)
    if total_chars > max_chars:
        raise LLMCompanyMatchError(
            f"Company prompt static contract exceeds configured budget: {total_chars}>{max_chars}"
        )
    record_event(
        "company.prompt_budget",
        stage="company_matcher",
        status="ok",
        budget_chars=max_chars,
        total_chars=total_chars,
        impact_count=len(industry_impacts),
        candidate_count=len(compact_candidates),
        candidate_truncated_count=max(original_candidate_count - len(compact_candidates), 0),
        web_evidence_count=len(compact_web),
    )
    return prompt


def _compact_impact_for_prompt(impact: dict[str, Any], index: int, text_limit: int) -> dict[str, Any]:
    return {
        "impact_id": str(impact.get("impact_id") or f"IMP-{index:03d}"),
        "industry": _clip_prompt(impact.get("industry"), text_limit),
        "policy_measure": _clip_prompt(impact.get("policy_measure"), text_limit),
        "implementation_action": _clip_prompt(impact.get("implementation_action"), text_limit),
        "chain_segment": _clip_prompt(impact.get("chain_segment"), text_limit),
        "transmission_logic": _clip_prompt(impact.get("transmission_logic"), text_limit),
        "business_variables": [_clip_prompt(value, text_limit) for value in (impact.get("business_variables") or [])[:4]],
        "affected_company_types": [_clip_prompt(value, text_limit) for value in (impact.get("affected_company_types") or [])[:3]],
        "conditions": [_clip_prompt(value, text_limit) for value in (impact.get("conditions") or [])[:2]],
        "risks": [_clip_prompt(value, text_limit) for value in (impact.get("risks") or [])[:2]],
    }


def _compact_candidate_for_prompt(
    record: dict[str, Any],
    text_limit: int,
    evidence_limit: int,
) -> dict[str, Any]:
    provenance: list[dict[str, Any]] = []
    seen_impacts: set[str] = set()
    for item in record.get("provenance") or []:
        if not isinstance(item, dict):
            continue
        impact_id = str(item.get("impact_id") or "")
        if impact_id and impact_id in seen_impacts:
            continue
        if impact_id:
            seen_impacts.add(impact_id)
        provenance_item = {
            "impact_id": impact_id,
            "tool": str(item.get("tool") or ""),
            "tool_call_id": str(item.get("tool_call_id") or ""),
            "query": _clip_prompt(item.get("keyword") or item.get("sector"), 40),
        }
        if text_limit > 80:
            provenance_item.update(
                {
                    "source_type": str(item.get("source_type") or ""),
                    "react_step": item.get("react_step"),
                }
            )
        provenance.append(provenance_item)
        if len(provenance) >= 8:
            break
    evidence = []
    for item in (record.get("mcp_evidence") or [])[:evidence_limit]:
        if not isinstance(item, dict):
            continue
        evidence.append(
            {
                "source": _clip_prompt(item.get("source_org") or item.get("source_name"), 60),
                "date": _clip_prompt(item.get("published_date") or item.get("data_date") or "unknown", 24),
                "url": _clip_prompt(item.get("source_url"), 160),
                "text": _clip_prompt(item.get("summary") or item.get("text") or item.get("title"), text_limit),
                "tool": str(item.get("tool_name") or ""),
            }
        )
    business_evidence = _clip_prompt(
        " ".join(
            _unique(
                [
                    str(record.get("matched_business") or ""),
                    str(record.get("business_evidence") or ""),
                ]
            )
        ),
        text_limit,
    )
    compact = {
        "company_name": str(record.get("company_name") or ""),
        "stock_code": str(record.get("stock_code") or ""),
        "impact_ids": list(record.get("impact_ids") or sorted(seen_impacts)),
        "chain_segment": _clip_prompt(record.get("chain_segment"), text_limit),
        "business_evidence": business_evidence,
        "negative_evidence": [_clip_prompt(value, text_limit) for value in (record.get("negative_evidence") or [])[:2]],
        "data_date": _clip_prompt(record.get("data_date") or "unknown", 24),
        "provenance": provenance,
    }
    if text_limit > 80:
        compact.update(
            {
                "industry_segment": _clip_prompt(record.get("industry_segment"), text_limit),
                "revenue_relevance": _clip_prompt(record.get("revenue_relevance") or "unknown", 40),
                "source_url": _clip_prompt(record.get("source_url"), 160),
                "business_sources": evidence,
            }
        )
    return compact


def _compact_web_evidence_for_prompt(item: dict[str, Any], text_limit: int) -> dict[str, Any]:
    return {
        "title": _clip_prompt(item.get("title"), 100),
        "source_org": _clip_prompt(item.get("source_org"), 60),
        "published_date": _clip_prompt(item.get("published_date") or "unknown", 24),
        "source_url": _clip_prompt(item.get("source_url"), 160),
        "summary": _clip_prompt(item.get("summary"), text_limit),
        "query": _clip_prompt(item.get("query"), 100),
        "tool_name": str(item.get("tool_name") or ""),
        "impact_id": str(item.get("impact_id") or ""),
    }


def _retrieval_uncertainties(retrieval_by_impact: dict[str, dict[str, Any]]) -> list[str]:
    statuses = {str(item.get("status") or "empty") for item in retrieval_by_impact.values()}
    output: list[str] = []
    if "error" in statuses:
        output.append("CNFinancial 候选查询失败，未能判断部分路径是否存在候选；查询失败不等于真实返回空。")
    if "unavailable" in statuses:
        output.append("CNFinancial 候选工具不可用或已熔断，未完成部分路径的候选检索。")
    if statuses and statuses <= {"empty"}:
        output.append("CNFinancial 候选查询成功但真实返回空，暂未形成可用于业务匹配的候选公司。")
    return output or ["未形成可用于业务匹配的 CNFinancial A 股候选公司。"]


def _company_prompt_max_chars() -> int:
    raw = os.getenv("POLICYCHAIN_COMPANY_PROMPT_MAX_CHARS", "")
    try:
        return max(int(raw), 8000) if raw.strip() else DEFAULT_COMPANY_PROMPT_MAX_CHARS
    except ValueError:
        return DEFAULT_COMPANY_PROMPT_MAX_CHARS


def _json_compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _prompt_char_count(prompt: dict[str, str]) -> int:
    return len(prompt.get("system") or "") + len(prompt.get("user") or "")


def _clip_prompt(value: object, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(max_chars - 1, 1)].rstrip() + "…"


def _merge_external_evidence(existing: list[dict[str, Any]], new_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in [*existing, *new_items]:
        if str(item.get("tool_name") or "") in {"get_industry_list", "get_concept_list"}:
            continue
        key = (
            str(item.get("server_name") or item.get("stock_code") or ""),
            str(item.get("tool_name") or item.get("data_date") or ""),
            str(item.get("source_url") or item.get("title") or item.get("company_name") or ""),
        )
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _merge_tool_logs(existing: list[dict[str, Any]], new_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in [*existing, *new_items]:
        key = (
            str(item.get("server_name") or ""),
            str(item.get("tool_name") or ""),
            json.dumps(item.get("arguments") or {}, ensure_ascii=False, sort_keys=True),
        )
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _company_react_query(industry_impacts: list[dict[str, Any]], company_records: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for impact in industry_impacts:
        parts.extend(
            str(impact.get(key) or "")
            for key in ("industry", "chain_segment", "transmission_logic", "affected_company_types")
        )
    for company in company_records[:5]:
        parts.extend(str(company.get(key) or "") for key in ("company_name", "stock_code", "matched_business"))
    return " ".join(part for part in parts if part).strip()[:500] or "A-share company business evidence"


def _company_react_failure_reason(exc: Exception) -> str:
    message = str(exc).lower()
    if "valid json" in message or "json object" in message:
        return "planner_invalid_json"
    if "schema" in message or "validation" in message:
        return "planner_schema_error"
    if "tool" in message:
        return "optional_react_tool_error"
    return "optional_react_error"


def _tag_react_traces(
    stage: str,
    traces: list[dict[str, object]],
    impact_id: str = "",
) -> list[dict[str, object]]:
    return [{"stage": stage, "impact_id": impact_id, **trace} for trace in traces]


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value)).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output
