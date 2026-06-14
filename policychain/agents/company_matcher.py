from __future__ import annotations

from typing import Any

from policychain.mcp import MCPToolInvoker, consume_mcp_invoker_errors, is_unavailable_invoker
from policychain.safety import assert_no_investment_advice
from policychain.schemas.agent_outputs import CompanyEvidence, CompanyMatch, CompanyMatchOutput
from policychain.state import PolicyResearchState
from policychain.tools import (
    collect_annual_report_evidence,
    collect_company_candidates,
    collect_company_web_evidence,
    read_company_source,
    search_company_information,
)
from policychain.tools.mcp_tools import mcp_unavailable_uncertainty


MATCH_LEVELS = ("high", "medium", "low")
ANNUAL_REPORT_NOT_FOUND = "未在最近两期年报中找到充分证据"


class CompanyMatchError(RuntimeError):
    """Raised when Company Matcher cannot produce a structured result."""


def run_company_matcher(
    state: PolicyResearchState,
    top_k_per_industry: int = 1,
    mcp_invoker: MCPToolInvoker | None = None,
    use_annual_reports: bool = True,
) -> CompanyMatchOutput:
    """Run deterministic company-business matching from industry impacts."""

    output = match_companies_for_impacts(
        industry_impacts=state.industry_impacts,
        top_k_per_industry=top_k_per_industry,
        mcp_invoker=mcp_invoker,
        use_annual_reports=use_annual_reports,
    )
    payload = output.to_dict()
    state.company_candidates = getattr(output, "_candidate_records", payload["companies"])
    state.company_matches = payload["companies"]
    state.annual_report_evidence = getattr(output, "_annual_report_evidence", [])
    state.company_research = getattr(output, "_company_research", [])
    state.external_evidence = _merge_external_evidence(
        state.external_evidence,
        [*state.company_research, *state.annual_report_evidence],
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
    state.uncertainties = _unique([*state.uncertainties, *payload["uncertainties"]])
    return output


def match_companies_for_impacts(
    industry_impacts: list[dict[str, Any]],
    top_k_per_industry: int = 1,
    mcp_invoker: MCPToolInvoker | None = None,
    use_annual_reports: bool = True,
) -> CompanyMatchOutput:
    if not industry_impacts:
        return CompanyMatchOutput(uncertainties=["缺少行业影响分析，无法生成公司业务匹配清单。"])

    candidate_records = collect_company_candidates(
        industry_impacts=industry_impacts,
        invoker=mcp_invoker,
        top_k_per_industry=top_k_per_industry,
    )
    using_mcp_candidates = bool(candidate_records)
    annual_report_evidence = (
        collect_annual_report_evidence(candidate_records, industry_impacts, invoker=mcp_invoker)
        if using_mcp_candidates and use_annual_reports
        else []
    )
    company_research = (
        collect_company_web_evidence(candidate_records, invoker=mcp_invoker)
        if using_mcp_candidates
        else []
    )

    matches: list[CompanyMatch] = []
    for impact in industry_impacts:
        industry = str(impact.get("industry") or "")
        keywords = _impact_keywords(impact)
        company_records = (
            candidate_records
            if using_mcp_candidates
            else search_company_information(
                industry_segment=industry,
                keywords=keywords,
                top_k=top_k_per_industry,
            )
        )
        for record in company_records:
            matches.append(
                _build_company_match(
                    record,
                    impact,
                    annual_evidence=_annual_evidence_for_company(annual_report_evidence, record),
                    require_annual_report_evidence=using_mcp_candidates and use_annual_reports,
                )
            )

    uncertainties = [
        "公司资料来自 MCP 候选和年报证据接口；若 MCP 未配置则回退到本地 mock 数据验证流程。",
        "匹配结果表示业务相关性，不构成任何投资建议。",
    ]
    if using_mcp_candidates and not use_annual_reports:
        uncertainties.append("CNINFO 年报下载本次已跳过；公司匹配仅使用 CNFinancial 候选数据和公开网页证据，业务真实性需后续年报验证。")
    output = CompanyMatchOutput(companies=_dedupe_matches(matches), uncertainties=uncertainties)
    setattr(output, "_candidate_records", candidate_records if using_mcp_candidates else [company.to_dict() for company in output.companies])
    setattr(output, "_annual_report_evidence", annual_report_evidence)
    setattr(output, "_company_research", company_research)
    assert_no_investment_advice(output.to_dict(), context="Company match output")
    return output


def _build_company_match(
    record: dict[str, Any],
    impact: dict[str, Any],
    annual_evidence: list[dict[str, Any]] | None = None,
    require_annual_report_evidence: bool = False,
) -> CompanyMatch:
    source = read_company_source(record)
    confidence = _confidence(record, impact)
    match_level = _match_level(confidence)
    industry = str(impact.get("industry") or record.get("industry_segment") or "")
    policy_link = str(impact.get("transmission_logic") or "")
    evidence = CompanyEvidence(
        source_name=str(source.get("source_name") or "mock company source"),
        source_url=source.get("source_url"),
        text=str(source.get("text") or ""),
        data_date=str(source.get("data_date") or ""),
    )
    annual_company_evidence = _company_evidence_from_annual_reports(annual_evidence or [])
    negative_evidence = _negative_evidence(annual_evidence or [], require_annual_report_evidence)
    if negative_evidence:
        match_level = "low"
        confidence = min(confidence, 0.45)
    return CompanyMatch(
        company_name=str(record.get("company_name") or ""),
        stock_code=str(record.get("stock_code") or ""),
        industry_segment=industry,
        chain_segment=str(impact.get("chain_segment") or industry),
        matched_business=str(record.get("matched_business") or ""),
        related_product_or_business=str(record.get("matched_business") or ""),
        match_level=match_level,
        annual_report_evidence=annual_company_evidence,
        revenue_or_ratio=str(record.get("revenue_or_ratio") or record.get("revenue_relevance") or ""),
        source_url=source.get("source_url"),
        match_conditions=list(impact.get("conditions") or []),
        negative_evidence=negative_evidence,
        business_evidence=[evidence],
        policy_link=policy_link,
        revenue_relevance=str(record.get("revenue_relevance") or "unknown"),
        conditions=list(impact.get("conditions") or []),
        risks=[*list(impact.get("risks") or []), *negative_evidence],
        data_date=str(record.get("data_date") or ""),
        confidence=confidence,
    )


def _confidence(record: dict[str, Any], impact: dict[str, Any]) -> float:
    industry = str(impact.get("industry") or "")
    keywords = list(record.get("business_keywords") or [])
    logic = str(impact.get("transmission_logic") or "")
    score = 0.45
    if industry and industry == record.get("industry_segment"):
        score += 0.25
    score += min(sum(1 for keyword in keywords if keyword in logic or keyword in industry) * 0.08, 0.24)
    return round(min(score, 0.95), 2)


def _match_level(confidence: float) -> str:
    if confidence >= 0.78:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


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


def _dedupe_matches(matches: list[CompanyMatch]) -> list[CompanyMatch]:
    seen: set[tuple[str, str]] = set()
    output: list[CompanyMatch] = []
    for match in matches:
        key = (match.company_name, match.industry_segment)
        if key not in seen:
            seen.add(key)
            output.append(match)
    return output


def _annual_evidence_for_company(evidence: list[dict[str, Any]], record: dict[str, Any]) -> list[dict[str, Any]]:
    name = str(record.get("company_name") or "")
    code = str(record.get("stock_code") or "")
    return [
        item
        for item in evidence
        if str(item.get("company_name") or "") == name or (code and str(item.get("stock_code") or "") == code)
    ]


def _company_evidence_from_annual_reports(items: list[dict[str, Any]]) -> list[CompanyEvidence]:
    evidence: list[CompanyEvidence] = []
    for item in items:
        evidence.append(
            CompanyEvidence(
                source_name=str(item.get("title") or "CNINFO annual report"),
                source_url=item.get("source_url"),
                text=str(item.get("text") or ""),
                data_date=str(item.get("report_year") or ""),
                report_year=item.get("report_year"),
                revenue_or_ratio=str(item.get("revenue_or_ratio") or ""),
                evidence_found=bool(item.get("evidence_found")),
            )
        )
    return evidence


def _negative_evidence(items: list[dict[str, Any]], require_annual_report_evidence: bool) -> list[str]:
    if not require_annual_report_evidence:
        return []
    if not items or not any(item.get("evidence_found") for item in items):
        return [ANNUAL_REPORT_NOT_FOUND]
    return [str(item.get("text")) for item in items if item.get("evidence_found") is False and item.get("text")]


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


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output
