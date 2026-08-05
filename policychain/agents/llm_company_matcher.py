from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import uuid4

from policychain.agents.company_matcher import audit_company_match_output, match_candidate_records_to_impacts, resolve_company_match_limit
from policychain.llm import LLMClient, create_llm_client, observed_llm_generate
from policychain.mcp import MCPToolInvoker, consume_mcp_invoker_errors, is_unavailable_invoker, mcp_server_is_unavailable
from policychain.observability import record_event
from policychain.prompts import render_prompt
from policychain.schemas.agent_outputs import CompanyMatchOutput, CompanySeedOutput
from policychain.state import PolicyResearchState
from policychain.structured_output import parse_structured_output
from policychain.tools import (
    collect_company_candidates,
    collect_company_web_evidence,
    merge_react_company_candidates,
    run_company_react_search,
)
from policychain.tools.mcp_tools import (
    CNFINANCIAL_SERVER,
    candidate_retrieval_statuses,
    mcp_unavailable_uncertainty,
    resolve_company_seeds,
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
    setattr(state, "company_seeds", seed_records)
    setattr(state, "company_seed_audit", [*cached_seed_audit, *resolved_seed_audit])
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    identity_verified = [record for record in company_records if _record_identity_is_verified(record)]
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
        record_event(
            "company.rank",
            stage="company_matcher",
            status=status,
            seed_id=str(provenance.get("seed_id") or ""),
            impact_id=key[0],
            tool_call_id=str(provenance.get("tool_call_id") or ""),
            reason_code=reason_code or decision,
            cache_hit=False,
            source="deterministic_company_audit",
            url=str(record.get("source_url") or ""),
            date=str(record.get("data_date") or ""),
            company_name=key[1],
            stock_code=key[2],
            rank=included_rank,
            confidence=float(audit.get("confidence") or 0.0),
        )
    return ranked, coverage, audit_logs


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
