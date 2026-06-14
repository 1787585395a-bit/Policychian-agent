from __future__ import annotations

import json
from typing import Any, Iterable

from policychain.llm import LLMClient, create_llm_client
from policychain.mcp import MCPToolInvoker, consume_mcp_invoker_errors, is_unavailable_invoker
from policychain.prompts import render_prompt
from policychain.schemas.agent_outputs import CompanyMatchOutput
from policychain.state import PolicyResearchState
from policychain.structured_output import parse_structured_output
from policychain.tools import (
    collect_annual_report_evidence,
    collect_company_candidates,
    collect_company_web_evidence,
    search_company_information,
)
from policychain.tools.mcp_tools import mcp_unavailable_uncertainty


class LLMCompanyMatchError(RuntimeError):
    """Raised when LLM Company Matcher output cannot be trusted."""


def run_llm_company_matcher(
    state: PolicyResearchState,
    llm_client: LLMClient | None = None,
    top_k_per_industry: int = 1,
    mcp_invoker: MCPToolInvoker | None = None,
    use_annual_reports: bool = True,
) -> CompanyMatchOutput:
    """Run the optional LLM Company Matcher over retrieved company candidates."""

    if not state.industry_impacts:
        output = CompanyMatchOutput(
            uncertainties=["缺少行业影响分析，无法生成公司业务匹配清单。"]
        )
        _write_state(state, output, candidates=[])
        return output

    company_records = _company_records_for_impacts(
        state.industry_impacts,
        top_k_per_industry=top_k_per_industry,
        mcp_invoker=mcp_invoker,
    )
    if not company_records:
        output = CompanyMatchOutput(
            uncertainties=["未从本地公司资料中检索到可用于业务匹配的候选公司。"]
        )
        _write_state(state, output, candidates=[])
        return output

    annual_report_evidence = (
        collect_annual_report_evidence(
            company_records,
            state.industry_impacts,
            invoker=mcp_invoker,
        )
        if use_annual_reports
        else []
    )
    company_web_evidence = collect_company_web_evidence(company_records, invoker=mcp_invoker)
    state.annual_report_evidence = annual_report_evidence
    state.company_research = company_web_evidence
    state.external_evidence = _merge_external_evidence(
        state.external_evidence,
        [*annual_report_evidence, *company_web_evidence],
    )
    if is_unavailable_invoker(mcp_invoker):
        state.uncertainties = _unique(
            [
                *state.uncertainties,
                mcp_unavailable_uncertainty("CNFinancial"),
                mcp_unavailable_uncertainty("CNINFO"),
                mcp_unavailable_uncertainty("Open-WebSearch"),
            ]
        )
    state.uncertainties = _unique([*state.uncertainties, *consume_mcp_invoker_errors(mcp_invoker)])
    if company_records and not use_annual_reports:
        state.uncertainties = _unique(
            [
                *state.uncertainties,
                "CNINFO 年报下载本次已跳过；LLM Company Matcher 仅使用候选公司数据和公开网页证据。",
            ]
        )

    prompt = render_prompt(
        "company_matcher",
        industry_impacts=_json_for_prompt(state.industry_impacts),
        company_records=_json_for_prompt(company_records),
        annual_report_evidence=_json_for_prompt(annual_report_evidence),
        web_evidence=_json_for_prompt(company_web_evidence),
    )
    client = llm_client or create_llm_client()
    raw_output = client.generate(prompt["system"], prompt["user"])
    output = parse_structured_output(raw_output, prompt["output_schema_name"])
    if not isinstance(output, CompanyMatchOutput):
        raise LLMCompanyMatchError("LLM Company Matcher returned an unexpected output schema")
    _assert_company_names_from_candidates(output, company_records)
    _write_state(state, output, candidates=company_records)
    return output


def _company_records_for_impacts(
    industry_impacts: list[dict[str, Any]],
    top_k_per_industry: int,
    mcp_invoker: MCPToolInvoker | None = None,
) -> list[dict[str, Any]]:
    mcp_records = collect_company_candidates(
        industry_impacts,
        invoker=mcp_invoker,
        top_k_per_industry=top_k_per_industry,
    )
    if mcp_records:
        return mcp_records

    records: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for impact in industry_impacts:
        industry = str(impact.get("industry") or "")
        keywords = _impact_keywords(impact)
        for record in search_company_information(industry_segment=industry, keywords=keywords, top_k=top_k_per_industry):
            company_name = str(record.get("company_name") or "")
            if company_name and company_name not in seen_names:
                seen_names.add(company_name)
                records.append(record)
    return records


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
) -> None:
    payload = output.to_dict()
    state.company_candidates = list(candidates)
    state.company_matches = payload["companies"]
    state.uncertainties = _unique([*state.uncertainties, *payload["uncertainties"]])


def _impact_keywords(impact: dict[str, Any]) -> list[str]:
    values = [
        str(impact.get("industry") or ""),
        str(impact.get("transmission_logic") or ""),
        " ".join(impact.get("conditions") or []),
        " ".join(impact.get("risks") or []),
    ]
    text = " ".join(values)
    candidates = (
        "生成式人工智能",
        "服务提供者",
        "算法",
        "模型",
        "训练数据",
        "数据",
        "个人信息",
        "安全评估",
        "内容",
        "标识",
        "未成年人",
        "合规",
    )
    return [candidate for candidate in candidates if candidate in text]


def _json_for_prompt(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _merge_external_evidence(existing: list[dict[str, Any]], new_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in [*existing, *new_items]:
        key = (
            str(item.get("server_name") or item.get("stock_code") or ""),
            str(item.get("tool_name") or item.get("report_year") or ""),
            str(item.get("source_url") or item.get("title") or item.get("company_name") or ""),
        )
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output
