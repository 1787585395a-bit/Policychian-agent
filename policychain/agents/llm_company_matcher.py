from __future__ import annotations

import json
import re
from typing import Any, Iterable

from policychain.agents.company_matcher import audit_company_match_output, match_candidate_records_to_impacts
from policychain.llm import LLMClient, create_llm_client
from policychain.mcp import MCPToolInvoker, consume_mcp_invoker_errors, is_unavailable_invoker
from policychain.prompts import render_prompt
from policychain.schemas.agent_outputs import CompanyMatchOutput
from policychain.state import PolicyResearchState
from policychain.structured_output import parse_structured_output
from policychain.tools import (
    collect_company_candidates,
    collect_company_web_evidence,
    run_company_react_search,
)
from policychain.tools.mcp_tools import mcp_unavailable_uncertainty


class LLMCompanyMatchError(RuntimeError):
    """Raised when LLM Company Matcher output cannot be trusted."""


def run_llm_company_matcher(
    state: PolicyResearchState,
    llm_client: LLMClient | None = None,
    top_k_per_industry: int = 3,
    mcp_invoker: MCPToolInvoker | None = None,
) -> CompanyMatchOutput:
    """Run the optional LLM Company Matcher over retrieved company candidates."""

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
    state.tool_call_logs.extend(tool_logs)
    if not company_records:
        output = CompanyMatchOutput(uncertainties=["未从 CNFinancial/Web 检索到可用于业务匹配的候选公司。"])
        _matches, coverage, audit_logs = match_candidate_records_to_impacts(
            state.industry_impacts,
            [],
            top_k_per_industry=top_k_per_industry,
        )
        _write_state(state, output, candidates=[], coverage=coverage, audit_logs=audit_logs)
        return output

    client = llm_client or create_llm_client()
    company_web_evidence = collect_company_web_evidence(company_records, invoker=mcp_invoker, tool_logs=tool_logs)
    state.tool_call_logs = _merge_tool_logs(state.tool_call_logs, tool_logs)
    if not is_unavailable_invoker(mcp_invoker):
        react_query = _company_react_query(state.industry_impacts, company_records)
        react_run = run_company_react_search(react_query, invoker=mcp_invoker, llm_client=client)
        state.react_traces.extend(_tag_react_traces("company", react_run.traces))
        company_web_evidence = _merge_external_evidence(company_web_evidence, react_run.evidence)
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

    prompt = render_prompt(
        "company_matcher",
        industry_impacts=_json_for_prompt(state.industry_impacts),
        company_records=_json_for_prompt(company_records),
        web_evidence=_json_for_prompt(company_web_evidence),
    )
    raw_output = client.generate(prompt["system"], prompt["user"])
    output = parse_structured_output(raw_output, prompt["output_schema_name"])
    if not isinstance(output, CompanyMatchOutput):
        raise LLMCompanyMatchError("LLM Company Matcher returned an unexpected output schema")
    _assert_company_names_from_candidates(output, company_records)
    audited_output = audit_company_match_output(
        output,
        state.industry_impacts,
        company_records,
        top_k_per_industry=top_k_per_industry,
    )
    _write_state(
        state,
        audited_output,
        candidates=company_records,
        coverage=getattr(audited_output, "_company_coverage", []),
        audit_logs=getattr(audited_output, "_audit_logs", []),
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


def _assert_company_names_from_candidates(
    output: CompanyMatchOutput,
    company_records: list[dict[str, Any]],
) -> None:
    allowed_names = {str(record.get("company_name") or "") for record in company_records}
    mismatches = sorted(company.company_name for company in output.companies if company.company_name not in allowed_names)
    if mismatches:
        raise LLMCompanyMatchError(
            f"LLM Company Matcher returned company outside candidate records: {', '.join(mismatches)}"
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


def _json_for_prompt(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _merge_external_evidence(existing: list[dict[str, Any]], new_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in [*existing, *new_items]:
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


def _tag_react_traces(stage: str, traces: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{"stage": stage, **trace} for trace in traces]


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value)).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output
