from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import os
import re
from typing import Any

from policychain.mcp import MCPToolInvoker, consume_mcp_invoker_errors, is_unavailable_invoker
from policychain.observability import record_event
from policychain.safety import assert_no_investment_advice
from policychain.schemas.agent_outputs import CompanyEvidence, CompanyMatch, CompanyMatchOutput
from policychain.state import PolicyResearchState
from policychain.tools import collect_company_candidates, collect_company_web_evidence, read_company_source
from policychain.tools.mcp_tools import candidate_retrieval_statuses, mcp_unavailable_uncertainty


MATCH_LEVELS = ("high", "medium", "low")
DEFAULT_MAX_COMPANIES_PER_IMPACT = 3
MAX_COMPANIES_PER_IMPACT = 3

DOMAIN_TERMS = (
    "钢铁",
    "冶炼",
    "轧钢",
    "节能",
    "环保",
    "绿色",
    "低碳",
    "能源",
    "新能源",
    "电池",
    "光伏",
    "风电",
    "储能",
    "汽车",
    "充电",
    "算力",
    "智算",
    "数据中心",
    "IDC",
    "云计算",
    "工业互联网",
    "数字化",
    "数智",
    "人工智能",
    "大模型",
    "模型",
    "算法",
    "数据",
    "安全",
    "合规",
    "评估",
    "软件",
    "平台",
    "通信",
    "设备",
    "服务器",
    "芯片",
    "半导体",
    "机器人",
    "低空",
    "航空",
    "材料",
    "研发",
    "服务",
    "制造",
)

WEAK_SOURCE_TOOLS = {"search_stock"}
GENERIC_SHARED_TERMS = {
    "能源",
    "新能源",
    "服务",
    "制造",
    "电力",
    "企业",
    "行业",
    "产业",
    "公司",
    "业务",
    "产品",
    "设备",
    "平台",
    "技术",
    "综合服务",
    "技术服务",
    "电力服务",
    "设备制造",
    "产品服务",
}
GENERIC_SHARED_TERMS_COMPACT = {re.sub(r"\s+", "", term).lower() for term in GENERIC_SHARED_TERMS}


class CompanyMatchError(RuntimeError):
    """Raised when Company Matcher cannot produce a structured result."""


def run_company_matcher(
    state: PolicyResearchState,
    top_k_per_industry: int = DEFAULT_MAX_COMPANIES_PER_IMPACT,
    mcp_invoker: MCPToolInvoker | None = None,
) -> CompanyMatchOutput:
    """Run deterministic company-business matching from industry impacts."""

    top_k_per_industry = resolve_company_match_limit(top_k_per_industry)

    if resolve_company_discovery_mode() == "web_first":
        output = CompanyMatchOutput(
            uncertainties=[
                "默认 Web-first 公司发现需要 DeepSeek discovery；当前确定性/fallback 路径未调用旧 CNFinancial-first 候选召回。",
                "公司部分仅表示业务相关性研究清单，不构成任何投资建议。",
            ]
        )
        coverage = _build_web_first_unavailable_coverage(state.industry_impacts)
        payload = output.to_dict()
        state.company_candidates = []
        state.company_matches = []
        state.company_coverage = coverage
        state.company_match_audit = []
        state.uncertainties = _unique([*state.uncertainties, *payload["uncertainties"]])
        return output

    output = match_companies_for_impacts(
        industry_impacts=state.industry_impacts,
        top_k_per_industry=top_k_per_industry,
        mcp_invoker=mcp_invoker,
    )
    payload = output.to_dict()
    state.company_candidates = getattr(output, "_candidate_records", payload["companies"])
    state.company_matches = payload["companies"]
    state.company_coverage = getattr(output, "_company_coverage", [])
    state.company_match_audit = getattr(output, "_audit_logs", [])
    state.company_research = getattr(output, "_company_research", [])
    state.tool_call_logs.extend(getattr(output, "_tool_call_logs", []))
    state.external_evidence = _merge_external_evidence(
        state.external_evidence,
        state.company_research,
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
    state.uncertainties = _unique([*state.uncertainties, *payload["uncertainties"]])
    return output


def match_companies_for_impacts(
    industry_impacts: list[dict[str, Any]],
    top_k_per_industry: int = DEFAULT_MAX_COMPANIES_PER_IMPACT,
    mcp_invoker: MCPToolInvoker | None = None,
) -> CompanyMatchOutput:
    top_k_per_industry = resolve_company_match_limit(top_k_per_industry)
    if not industry_impacts:
        return CompanyMatchOutput(uncertainties=["缺少行业影响分析，无法生成公司业务匹配清单。"])

    tool_logs: list[dict[str, Any]] = []
    candidate_records = collect_company_candidates(
        industry_impacts=industry_impacts,
        invoker=mcp_invoker,
        top_k_per_industry=top_k_per_industry,
        tool_logs=tool_logs,
    )
    company_research = (
        collect_company_web_evidence(candidate_records, invoker=mcp_invoker, tool_logs=tool_logs)
        if candidate_records
        else []
    )
    retrieval_by_impact = candidate_retrieval_statuses(tool_logs)

    companies, coverage, audit_logs = match_candidate_records_to_impacts(
        industry_impacts=industry_impacts,
        candidate_records=candidate_records,
        top_k_per_industry=top_k_per_industry,
        retrieval_by_impact=retrieval_by_impact,
    )
    tool_logs.extend(_audit_tool_logs(coverage))

    uncertainties = [
        "公司资料来自 CNFinancial/Web 候选证据；MCP 不可用或返回空时，不再使用本地 mock 公司填充用户报告。",
        "公司部分仅表示业务相关性研究清单，不构成任何投资建议。",
    ]
    if not candidate_records:
        uncertainties.extend(_candidate_retrieval_uncertainties(retrieval_by_impact))
    if candidate_records and not companies:
        uncertainties.append("CNFinancial 返回了候选公司，但未通过业务相关性审查；报告将按行业路径说明无可靠公司匹配原因。")

    output = CompanyMatchOutput(companies=companies, uncertainties=uncertainties)
    setattr(output, "_candidate_records", candidate_records)
    setattr(output, "_company_research", company_research)
    setattr(output, "_company_coverage", coverage)
    setattr(output, "_audit_logs", audit_logs)
    setattr(output, "_tool_call_logs", tool_logs)
    assert_no_investment_advice(output.to_dict(), context="Company match output")
    return output


def audit_company_match_output(
    output: CompanyMatchOutput,
    industry_impacts: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    top_k_per_industry: int = DEFAULT_MAX_COMPANIES_PER_IMPACT,
    retrieval_by_impact: dict[str, dict[str, Any]] | None = None,
    allow_weak_semantic_keep_low: bool = False,
) -> CompanyMatchOutput:
    """Bind LLM company matches to concrete impact paths and audit relevance."""

    top_k_per_industry = resolve_company_match_limit(top_k_per_industry)

    candidate_by_name = _candidate_index(candidate_records)
    audited: list[CompanyMatch] = []
    audit_logs: list[dict[str, Any]] = []
    matches_by_impact: dict[str, list[CompanyMatch]] = {}

    for match in output.companies:
        record = candidate_by_name.get(_company_key(match.company_name, match.stock_code))
        if not record:
            record = candidate_by_name.get(_company_key(match.company_name, ""))
        allowed_impact_ids = _candidate_impact_ids(record or {})
        if allowed_impact_ids and match.impact_id not in allowed_impact_ids:
            impact_index, impact = _impact_for_claimed_id(match.impact_id, industry_impacts)
            status = {
                "decision": "reject",
                "reason_code": "path_provenance_mismatch",
                "match_level": "low",
                "confidence": 0.0,
                "reason": (
                    f"LLM 输出路径 {match.impact_id or 'empty'} 不在候选检索来源路径中；"
                    f"允许路径为 {', '.join(sorted(allowed_impact_ids))}。"
                ),
                "shared_terms": [],
                "negative_evidence": ["候选公司不得从检索来源路径自动改绑到其他行业影响路径。"],
            }
            audited_match = _apply_audit(match, impact_index, impact, status)
            audit_logs.append(_audit_log_entry(audited_match, status, impact_index, impact))
            continue
        if allowed_impact_ids:
            impact_index, impact = _impact_for_claimed_id(match.impact_id, industry_impacts)
        else:
            impact_index, impact = _best_impact_for_match(match, record or {}, industry_impacts)
        status = _audit_candidate_against_impact(record or _company_match_as_record(match), impact)
        if allow_weak_semantic_keep_low:
            status = _weak_semantic_keep_low_status(match, record or {}, impact, status)
        audited_match = _apply_audit(match, impact_index, impact, status)
        audit_logs.append(_audit_log_entry(audited_match, status, impact_index, impact))
        if status["decision"] == "reject":
            continue
        matches_by_impact.setdefault(_impact_id(impact_index), []).append(audited_match)

    for impact_index, _impact in enumerate(industry_impacts, start=1):
        impact_id = _impact_id(impact_index)
        scoped = sorted(matches_by_impact.get(impact_id, []), key=lambda item: item.confidence, reverse=True)
        audited.extend(scoped[:top_k_per_industry])
        for trimmed in scoped[top_k_per_industry:]:
            _mark_company_audit_cap_trimmed(audit_logs, trimmed)

    coverage = _build_coverage_matrix(
        industry_impacts=industry_impacts,
        candidate_records=candidate_records,
        matches=audited,
        audit_logs=audit_logs,
        retrieval_by_impact=retrieval_by_impact,
    )
    audited_output = CompanyMatchOutput(companies=audited, uncertainties=list(output.uncertainties))
    setattr(audited_output, "_company_coverage", coverage)
    setattr(audited_output, "_audit_logs", audit_logs)
    return audited_output


def match_candidate_records_to_impacts(
    industry_impacts: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    top_k_per_industry: int = DEFAULT_MAX_COMPANIES_PER_IMPACT,
    retrieval_by_impact: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[CompanyMatch], list[dict[str, Any]], list[dict[str, Any]]]:
    top_k_per_industry = resolve_company_match_limit(top_k_per_industry)
    matches: list[CompanyMatch] = []
    audit_logs: list[dict[str, Any]] = []

    for impact_index, impact in enumerate(industry_impacts, start=1):
        scoped_impact = {**impact, "impact_id": str(impact.get("impact_id") or _impact_id(impact_index))}
        scoped: list[CompanyMatch] = []
        for record in candidate_records:
            if not _candidate_is_for_impact(record, _impact_id(impact_index)):
                continue
            status = _audit_candidate_against_impact(record, scoped_impact)
            provisional = _build_company_match(record, impact_index, scoped_impact, status)
            audit_logs.append(_audit_log_entry(provisional, status, impact_index, scoped_impact))
            if status["decision"] == "reject":
                continue
            scoped.append(provisional)

        scoped.sort(key=lambda item: item.confidence, reverse=True)
        matches.extend(scoped[:top_k_per_industry])

    deduped = _dedupe_matches(matches)
    coverage = _build_coverage_matrix(
        industry_impacts=industry_impacts,
        candidate_records=candidate_records,
        matches=deduped,
        audit_logs=audit_logs,
        retrieval_by_impact=retrieval_by_impact,
    )
    return deduped, coverage, audit_logs


def _build_company_match(
    record: dict[str, Any],
    impact_index: int,
    impact: dict[str, Any],
    audit_status: dict[str, Any] | None = None,
) -> CompanyMatch:
    source = read_company_source(record)
    audit_status = audit_status or _audit_candidate_against_impact(record, impact)
    confidence = float(audit_status["confidence"])
    match_level = str(audit_status["match_level"])
    industry = str(impact.get("industry") or record.get("industry_segment") or "")
    chain_segment = str(impact.get("chain_segment") or record.get("chain_segment") or industry)
    policy_link = str(impact.get("transmission_logic") or "")
    evidence = CompanyEvidence(
        source_name=str(source.get("source_name") or record.get("source_name") or "CNFinancial/Web"),
        source_url=source.get("source_url"),
        text=str(source.get("text") or record.get("business_evidence") or record.get("matched_business") or ""),
        data_date=str(source.get("data_date") or record.get("data_date") or "unknown"),
        revenue_or_ratio=str(record.get("revenue_or_ratio") or record.get("revenue_relevance") or ""),
    )
    negative_evidence = list(audit_status.get("negative_evidence") or [])
    return CompanyMatch(
        company_name=str(record.get("company_name") or ""),
        stock_code=str(record.get("stock_code") or ""),
        industry_segment=industry,
        impact_id=_impact_id(impact_index),
        impact_industry=industry,
        chain_segment=chain_segment,
        matched_business=str(record.get("matched_business") or record.get("business_evidence") or ""),
        related_product_or_business=str(record.get("matched_business") or record.get("business_evidence") or ""),
        match_level=match_level,
        revenue_or_ratio=str(record.get("revenue_or_ratio") or record.get("revenue_relevance") or ""),
        source_url=source.get("source_url"),
        match_conditions=list(impact.get("conditions") or []),
        negative_evidence=negative_evidence,
        business_evidence=[evidence] if evidence.text else [],
        policy_link=policy_link,
        revenue_relevance=str(record.get("revenue_relevance") or "unknown"),
        conditions=list(impact.get("conditions") or []),
        risks=[*list(impact.get("risks") or []), *negative_evidence],
        data_date=str(record.get("data_date") or source.get("data_date") or "unknown"),
        confidence=confidence,
        audit_status=str(audit_status["decision"]),
        audit_reason=str(audit_status["reason"]),
    )


def _audit_candidate_against_impact(record: dict[str, Any], impact: dict[str, Any]) -> dict[str, Any]:
    impact_text = _impact_context(impact)
    company_text = _company_context(record)
    impact_id = str(impact.get("impact_id") or "")
    web_fallback = bool(record.get("web_fallback_verified")) and (
        not record.get("web_fallback_impacts") or impact_id in (record.get("web_fallback_impacts") or [])
    )
    shared_terms = _shared_terms(company_text, impact_text)
    verified_terms = (record.get("verified_path_terms_by_impact") or {}).get(impact_id) or []
    shared_terms = _unique([*shared_terms, *verified_terms])
    specific_shared_terms = [term for term in shared_terms if not _is_generic_shared_term(term)]
    has_business_evidence = bool(str(record.get("business_evidence") or record.get("matched_business") or "").strip())
    source_tool = str(record.get("candidate_source_tool") or "")
    confidence = _confidence(record, impact, shared_terms, has_business_evidence)
    negative_evidence: list[str] = []

    if not record.get("company_name"):
        return {
            "decision": "reject",
            "reason_code": "missing_company_name",
            "match_level": "low",
            "confidence": 0.0,
            "reason": "候选记录缺少公司名称。",
            "shared_terms": [],
            "negative_evidence": ["候选记录缺少公司名称。"],
        }

    explicit_contradiction = _explicit_business_contradiction(record)
    if explicit_contradiction:
        return {
            "decision": "reject",
            "reason_code": "explicit_business_contradiction",
            "match_level": "low",
            "confidence": 0.0,
            "reason": "业务资料含有与该路径明确冲突或明确不相关的陈述，不能作为匹配保留。",
            "shared_terms": shared_terms,
            "negative_evidence": [explicit_contradiction],
        }

    if _is_web_only_candidate(record):
        return {
            "decision": "reject",
            "reason_code": "web_only_candidate",
            "match_level": "low",
            "confidence": min(confidence, 0.2),
            "reason": "候选仅来自 Web 结果，不能进入 CNFinancial A 股候选白名单。",
            "shared_terms": shared_terms,
            "negative_evidence": ["Web 资料只能补充已验证候选的业务证据，不能单独形成候选公司。"],
        }

    if not shared_terms:
        negative_evidence.append("未发现公司业务与该政策路径的产业链环节或经营变量存在明确交集。")
    if not has_business_evidence:
        negative_evidence.append("缺少主营业务、产品服务、公告或官网等业务证据。")
    if source_tool in WEAK_SOURCE_TOOLS and not has_business_evidence:
        negative_evidence.append("候选仅来自关键词搜索，缺少业务证据支撑。")

    if shared_terms and not specific_shared_terms:
        negative_evidence.append("公司与路径仅共享服务、制造、电力等泛化词，缺少路径特异产品或业务交集。")
        return {
            "decision": "reject",
            "reason_code": "generic_only",
            "match_level": "low",
            "confidence": min(confidence, 0.3),
            "reason": "服务、制造、电力等泛化词不能单独支撑公司业务匹配。",
            "shared_terms": shared_terms,
            "negative_evidence": negative_evidence,
        }

    if not shared_terms and not has_business_evidence:
        return {
            "decision": "reject",
            "reason_code": "missing_business_and_intersection",
            "match_level": "low",
            "confidence": min(confidence, 0.25),
            "reason": "仅有行业或概念线索，缺少业务交集和业务证据。",
            "shared_terms": shared_terms,
            "negative_evidence": negative_evidence,
        }

    if not has_business_evidence:
        return {
            "decision": "reject",
            "reason_code": "missing_business_evidence",
            "match_level": "low",
            "confidence": min(confidence, 0.3),
            "reason": "缺少可回溯的主营业务、产品服务、公告或官网业务资料。",
            "shared_terms": shared_terms,
            "negative_evidence": negative_evidence,
        }

    if not shared_terms:
        return {
            "decision": "reject",
            "reason_code": "no_shared_terms",
            "match_level": "low",
            "confidence": min(confidence, 0.35),
            "reason": "有业务资料但未能直接对应该行业路径，剔除该路径下的公司匹配。",
            "shared_terms": shared_terms,
            "negative_evidence": negative_evidence,
        }

    if source_tool in WEAK_SOURCE_TOOLS and len(shared_terms) < 2:
        return {
            "decision": "keep_low",
            "reason_code": "weak_source_keep_low",
            "match_level": "low",
            "confidence": min(confidence, 0.55),
            "reason": "公司来自关键词补充搜索，业务交集较弱，不能提升为中高置信。",
            "shared_terms": shared_terms,
            "negative_evidence": negative_evidence,
        }

    if web_fallback:
        negative_evidence.append("CNFinancial 未完成交叉验证；当前仅有两处独立 Web 证据。")
        return {
            "decision": "keep_low",
            "reason_code": "web_fallback",
            "match_level": "low",
            "confidence": min(confidence, 0.55),
            "reason": "CNFinancial 技术失败后由两处独立 Web 证据支持，固定保留为低置信匹配。",
            "shared_terms": shared_terms,
            "negative_evidence": negative_evidence,
        }

    return {
        "decision": "passed",
        "reason_code": "strong_business_intersection",
        "match_level": _match_level(confidence),
        "confidence": confidence,
        "reason": f"公司业务证据与路径关键词存在交集：{', '.join(shared_terms[:5])}。",
        "shared_terms": shared_terms,
        "negative_evidence": negative_evidence,
    }


def _weak_semantic_keep_low_status(
    match: CompanyMatch,
    record: dict[str, Any],
    impact: dict[str, Any],
    status: dict[str, Any],
) -> dict[str, Any]:
    if str(status.get("decision") or "") != "reject":
        return status
    if str(status.get("reason_code") or "") not in {"generic_only", "no_shared_terms"}:
        return status
    impact_id = str(impact.get("impact_id") or match.impact_id or "")
    business_text = str(
        (record.get("business_evidence_by_impact") or {}).get(impact_id)
        or record.get("business_evidence")
        or record.get("matched_business")
        or ""
    ).strip()
    provenance = [
        item
        for item in (record.get("provenance") or [])
        if isinstance(item, dict)
        and (not impact_id or str(item.get("impact_id") or "") == impact_id)
        and any(item.get(field) for field in ("tool", "tool_call_id", "source_type"))
    ]
    explicit_conflict = _explicit_business_contradiction(record, match.negative_evidence)
    if not business_text or not provenance or explicit_conflict or _is_web_only_candidate(record):
        return status
    configured_cap = record.get("confidence_cap")
    try:
        record_cap = float(configured_cap) if configured_cap not in (None, "") else 0.92
    except (TypeError, ValueError):
        record_cap = 0.92
    web_fallback = bool(record.get("web_fallback_verified")) and (
        not record.get("web_fallback_impacts")
        or impact_id in (record.get("web_fallback_impacts") or [])
    )
    confidence_cap = 0.40 if web_fallback else 0.45
    return {
        "decision": "keep_low",
        "reason_code": "web_weak_semantic_keep_low" if web_fallback else "weak_semantic_keep_low",
        "match_level": "low",
        "confidence": min(float(match.confidence), record_cap, confidence_cap),
        "reason": "模型在身份与路径白名单内选择该公司；存在可回溯业务资料且无明确反面冲突，但词面交集较弱，固定保留为低置信。",
        "shared_terms": list(status.get("shared_terms") or []),
        "negative_evidence": _unique(
            [
                *list(status.get("negative_evidence") or []),
                "业务与政策路径的词面交集较弱，仅保留为低置信研究线索。",
            ]
        ),
    }


def _explicit_business_contradiction(
    record: dict[str, Any],
    extra_negative_evidence: list[str] | None = None,
) -> str:
    impact_specific_business = " ".join(
        str(value)
        for value in (record.get("business_evidence_by_impact") or {}).values()
        if value
    )
    negative_text = " ".join(
        [
            impact_specific_business,
            *(str(value) for value in (record.get("negative_evidence") or [])),
            *(str(value) for value in (extra_negative_evidence or [])),
        ]
    )
    match = re.search(
        r"明确不相关|业务不匹配|与.{0,12}无关|不涉及|未开展|不包含|不包括|未布局|不生产|未从事|已退出|停止经营|终止经营|否认.{0,12}业务|不存在相关",
        negative_text,
    )
    return match.group(0) if match else ""


def _confidence(
    record: dict[str, Any],
    impact: dict[str, Any],
    shared_terms: list[str] | None = None,
    has_business_evidence: bool | None = None,
) -> float:
    impact_text = _impact_context(impact)
    company_text = _company_context(record)
    shared_terms = shared_terms if shared_terms is not None else _shared_terms(company_text, impact_text)
    has_business_evidence = bool(company_text.strip()) if has_business_evidence is None else has_business_evidence

    score = 0.32
    if str(impact.get("industry") or "") and str(impact.get("industry") or "") == str(record.get("industry_segment") or ""):
        score += 0.14
    if str(impact.get("chain_segment") or "") and str(impact.get("chain_segment") or "") in company_text:
        score += 0.16
    score += min(len(shared_terms) * 0.08, 0.28)
    if has_business_evidence:
        score += 0.1
    if record.get("revenue_relevance") not in (None, "", "unknown"):
        score += 0.05
    if record.get("candidate_source_tool") == "get_industry_stocks":
        score += 0.04
    configured_cap = record.get("confidence_cap")
    try:
        confidence_cap = min(float(configured_cap), 0.92) if configured_cap not in (None, "") else 0.92
    except (TypeError, ValueError):
        confidence_cap = 0.92
    return round(min(score, confidence_cap), 2)


def _apply_audit(
    match: CompanyMatch,
    impact_index: int,
    impact: dict[str, Any],
    status: dict[str, Any],
) -> CompanyMatch:
    negative_evidence = _unique([*match.negative_evidence, *list(status.get("negative_evidence") or [])])
    confidence = min(float(match.confidence), float(status["confidence"]))
    if match.match_level == "high" and status["match_level"] != "high":
        match_level = str(status["match_level"])
    elif status["decision"] != "passed":
        match_level = "low"
    else:
        match_level = _match_level(confidence)
    return CompanyMatch(
        company_name=match.company_name,
        stock_code=match.stock_code,
        industry_segment=str(impact.get("industry") or match.industry_segment),
        impact_id=_impact_id(impact_index),
        impact_industry=str(impact.get("industry") or match.impact_industry or match.industry_segment),
        chain_segment=str(impact.get("chain_segment") or match.chain_segment),
        matched_business=match.matched_business,
        related_product_or_business=match.related_product_or_business or match.matched_business,
        match_level=match_level,
        revenue_or_ratio=match.revenue_or_ratio,
        source_url=match.source_url,
        match_conditions=list(match.match_conditions),
        negative_evidence=negative_evidence,
        business_evidence=list(match.business_evidence),
        policy_link=match.policy_link or str(impact.get("transmission_logic") or ""),
        revenue_relevance=match.revenue_relevance,
        conditions=list(match.conditions),
        risks=_unique([*match.risks, *negative_evidence]),
        data_date=match.data_date or "unknown",
        confidence=round(confidence, 2),
        audit_status=str(status["decision"]),
        audit_reason=str(status["reason"]),
    )


def _build_coverage_matrix(
    industry_impacts: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    matches: list[CompanyMatch],
    audit_logs: list[dict[str, Any]],
    retrieval_by_impact: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    for index, impact in enumerate(industry_impacts, start=1):
        impact_id = _impact_id(index)
        scoped_matches = [match for match in matches if match.impact_id == impact_id]
        scoped_audits = [item for item in audit_logs if item.get("impact_id") == impact_id]
        scoped_candidates = [record for record in candidate_records if _candidate_is_for_impact(record, impact_id)]
        rejected_count = sum(1 for item in scoped_audits if item.get("decision") == "reject")
        cap_trimmed_count = sum(1 for item in scoped_audits if item.get("reason_code") == "cap_trimmed")
        retrieval = dict((retrieval_by_impact or {}).get(impact_id) or {})
        retrieval_status = str(retrieval.get("status") or ("ok" if scoped_candidates else "empty"))
        no_match_reason = ""
        if not scoped_candidates:
            if retrieval_status in {"unavailable", "cnfinancial_unavailable"}:
                no_match_reason = "CNFinancial 工具不可用或已熔断，未能完成该路径候选查询；这不等于真实返回空。"
            elif retrieval_status in {"error", "cnfinancial_error", "discovery_error"}:
                no_match_reason = "CNFinancial 候选查询失败，未能判断该路径是否存在候选；这不等于真实返回空。"
            else:
                no_match_reason = "CNFinancial 查询成功但真实返回空，暂未形成该路径的 A 股候选公司。"
        elif not scoped_matches:
            no_match_reason = "候选公司未通过业务相关性审查，或缺少主营业务/产品服务证据。"
        coverage.append(
            {
                "impact_id": impact_id,
                "industry": str(impact.get("industry") or ""),
                "policy_measure": str(impact.get("policy_measure") or ""),
                "implementation_action": str(impact.get("implementation_action") or ""),
                "chain_segment": str(impact.get("chain_segment") or ""),
                "business_variables": list(impact.get("business_variables") or []),
                "affected_company_types": list(impact.get("affected_company_types") or []),
                "candidate_count": len(scoped_candidates),
                "passed_count": len(scoped_matches),
                "final_count": len(scoped_matches),
                "rejected_count": rejected_count,
                "cap_trimmed_count": cap_trimmed_count,
                "company_names": [match.company_name for match in scoped_matches],
                "retrieval_status": retrieval_status,
                "retrieval_error": str(retrieval.get("error") or ""),
                "retrieval_queries": list(retrieval.get("query_terms") or []),
                "retrieval_query_count": int(retrieval.get("query_count") or 0),
                "retrieval_skipped_queries": list(retrieval.get("skipped_queries") or []),
                "retrieval_skipped_query_count": int(retrieval.get("skipped_query_count") or 0),
                "retrieval_channel_statuses": dict(retrieval.get("channel_statuses") or {}),
                "no_match_reason": no_match_reason,
            }
        )
    return coverage


def _audit_log_entry(
    match: CompanyMatch,
    status: dict[str, Any],
    impact_index: int,
    impact: dict[str, Any],
) -> dict[str, Any]:
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "impact_id": _impact_id(impact_index),
        "industry": str(impact.get("industry") or ""),
        "chain_segment": str(impact.get("chain_segment") or ""),
        "company_name": match.company_name,
        "stock_code": match.stock_code,
        "decision": status["decision"],
        "match_level": status["match_level"],
        "confidence": status["confidence"],
        "shared_terms": list(status.get("shared_terms") or []),
        "reason_code": str(status.get("reason_code") or ""),
        "reason": status["reason"],
    }
    record_event("rule.company_audit", stage="company_matcher", status=str(status["decision"]), **entry)
    return entry


def _mark_company_audit_cap_trimmed(
    audit_logs: list[dict[str, Any]],
    match: CompanyMatch,
) -> None:
    for entry in reversed(audit_logs):
        if (
            str(entry.get("impact_id") or "") == match.impact_id
            and str(entry.get("company_name") or "") == match.company_name
            and str(entry.get("stock_code") or "") == match.stock_code
            and str(entry.get("decision") or "") != "reject"
        ):
            entry["decision"] = "trimmed"
            entry["reason_code"] = "cap_trimmed"
            entry["reason"] = "该公司已通过证据审查，但因每条影响路径最多保留 3 家而未进入最终清单。"
            record_event(
                "rule.company_audit",
                stage="company_matcher",
                status="trimmed",
                **entry,
            )
            return


def _audit_tool_logs(coverage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_passed = sum(int(item.get("passed_count") or 0) for item in coverage)
    total_rejected = sum(int(item.get("rejected_count") or 0) for item in coverage)
    logs = [
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "server_name": "policychain",
            "tool_name": "company_match_audit",
            "arguments": {"impact_count": len(coverage)},
            "status": "ok",
            "count": total_passed,
            "error": "",
            "summary": f"covered={len(coverage)}, passed={total_passed}, rejected={total_rejected}",
        }
    ]
    for item in coverage:
        logs.append(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "server_name": "policychain",
                "tool_name": "company_match_coverage",
                "arguments": {
                    "impact_id": item.get("impact_id"),
                    "industry": item.get("industry"),
                    "chain_segment": item.get("chain_segment"),
                },
                "status": "ok" if item.get("passed_count") else "empty",
                "count": item.get("passed_count") or 0,
                "error": item.get("no_match_reason") or "",
                "summary": f"candidate_count={item.get('candidate_count')}, rejected={item.get('rejected_count')}",
            }
        )
    return logs


def _best_impact_for_match(
    match: CompanyMatch,
    record: dict[str, Any],
    industry_impacts: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    if not industry_impacts:
        return 1, {}
    if match.impact_id:
        for index, impact in enumerate(industry_impacts, start=1):
            if match.impact_id == _impact_id(index):
                return index, impact

    best_index = 1
    best_impact = industry_impacts[0]
    best_score = -1.0
    candidate_record = record or _company_match_as_record(match)
    for index, impact in enumerate(industry_impacts, start=1):
        scoped_impact = {**impact, "impact_id": str(impact.get("impact_id") or _impact_id(index))}
        status = _audit_candidate_against_impact(candidate_record, scoped_impact)
        score = float(status["confidence"])
        if status["decision"] == "reject":
            score -= 0.2
        if score > best_score:
            best_score = score
            best_index = index
            best_impact = impact
    return best_index, best_impact


def _impact_for_claimed_id(
    impact_id: str,
    industry_impacts: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    for index, impact in enumerate(industry_impacts, start=1):
        if impact_id == _impact_id(index):
            return index, impact
    return (1, industry_impacts[0]) if industry_impacts else (1, {})


def _company_match_as_record(match: CompanyMatch) -> dict[str, Any]:
    evidence_text = " ".join(item.text for item in match.business_evidence)
    return {
        "company_name": match.company_name,
        "stock_code": match.stock_code,
        "industry_segment": match.industry_segment,
        "chain_segment": match.chain_segment,
        "matched_business": match.matched_business or match.related_product_or_business,
        "business_evidence": evidence_text,
        "revenue_relevance": match.revenue_relevance,
        "data_date": match.data_date,
        "source_url": match.source_url,
    }


def _candidate_index(candidate_records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for record in candidate_records:
        name = str(record.get("company_name") or "")
        code = str(record.get("stock_code") or "")
        if not name:
            continue
        output[_company_key(name, code)] = record
        output.setdefault(_company_key(name, ""), record)
    return output


def _candidate_is_for_impact(record: dict[str, Any], impact_id: str) -> bool:
    impact_ids = _candidate_impact_ids(record)
    return not impact_ids or impact_id in impact_ids


def _candidate_impact_ids(record: dict[str, Any]) -> set[str]:
    impact_ids = {str(value) for value in record.get("impact_ids") or [] if value}
    impact_ids.update(
        str(item.get("impact_id") or "")
        for item in record.get("provenance") or []
        if isinstance(item, dict) and item.get("impact_id")
    )
    return impact_ids


def _company_key(name: str, code: str) -> tuple[str, str]:
    return (re.sub(r"\s+", "", name), re.sub(r"\s+", "", code))


def _impact_id(index: int) -> str:
    return f"IMP-{index:03d}"


def _impact_context(impact: dict[str, Any]) -> str:
    parts = [
        impact.get("industry"),
        impact.get("impact_type"),
        impact.get("direction"),
        impact.get("transmission_logic"),
        impact.get("policy_measure"),
        impact.get("implementation_action"),
        impact.get("chain_segment"),
        *(impact.get("business_variables") or []),
        *(impact.get("affected_company_types") or []),
        *(impact.get("conditions") or []),
        *(impact.get("risks") or []),
    ]
    return " ".join(str(part) for part in parts if part)


def _company_context(record: dict[str, Any]) -> str:
    evidence_items = record.get("mcp_evidence") or []
    evidence_text = " ".join(
        str(item.get("summary") or item.get("text") or item.get("title") or "")
        for item in evidence_items
        if isinstance(item, dict)
    )
    raw_payload = record.get("cnfinancial_raw") if isinstance(record.get("cnfinancial_raw"), dict) else {}
    parts = [
        record.get("company_name"),
        record.get("stock_code"),
        record.get("matched_business"),
        record.get("business_evidence"),
        record.get("revenue_relevance"),
        raw_payload.get("main_business") if raw_payload else "",
        raw_payload.get("description") if raw_payload else "",
        raw_payload.get("主营业务") if raw_payload else "",
        evidence_text,
    ]
    return " ".join(str(part) for part in parts if part)


def _shared_terms(company_text: str, impact_text: str) -> list[str]:
    company_compact = _compact_match_text(company_text)
    impact_compact = _compact_match_text(impact_text)
    terms: list[str] = []
    for term in DOMAIN_TERMS:
        compact = _compact_match_text(term)
        if compact and compact in company_compact and compact in impact_compact:
            terms.append(term)

    company_tokens = set(_match_tokens(company_text))
    impact_tokens = set(_match_tokens(impact_text))
    for token in sorted(company_tokens & impact_tokens):
        if len(token) >= 2 and token not in terms:
            terms.append(token)
    return _unique(terms)[:8]


def _is_generic_shared_term(value: str) -> bool:
    return _compact_match_text(value) in GENERIC_SHARED_TERMS_COMPACT


def _is_web_only_candidate(record: dict[str, Any]) -> bool:
    if record.get("web_fallback_verified") is True:
        return False
    provenance = [item for item in record.get("provenance") or [] if isinstance(item, dict)]
    explicit_sources: list[str] = []
    for item in provenance:
        explicit_sources.extend(
            [
                str(item.get("server_name") or "").lower(),
                str(item.get("source_type") or "").lower(),
                str(item.get("tool") or "").lower(),
            ]
        )
    source_tool = str(record.get("candidate_source_tool") or "").lower()
    source_type = str(record.get("source_type") or "").lower()
    explicit_sources.extend([source_tool, source_type])
    has_cnfinancial = any(
        "cnfinancial" in value
        or value in {"search_stock", "get_industry_stocks", "cn-financial"}
        for value in explicit_sources
    )
    has_web = any("web" in value or value in {"search", "fetchwebcontent", "web-search"} for value in explicit_sources)
    return bool(has_web and not has_cnfinancial)


def _compact_match_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).lower()


def _match_tokens(value: str) -> list[str]:
    return [
        token
        for token in re.split(r"[\s,，。；;、/|()（）:：]+", str(value))
        if len(token.strip()) >= 2
    ]


def _match_level(confidence: float) -> str:
    if confidence >= 0.78:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def resolve_company_match_limit(requested: int = DEFAULT_MAX_COMPANIES_PER_IMPACT) -> int:
    raw = os.getenv("POLICYCHAIN_MAX_COMPANY_MATCHES_PER_IMPACT")
    if raw and raw.strip():
        try:
            return min(max(int(raw), 1), MAX_COMPANIES_PER_IMPACT)
        except ValueError:
            pass
    return min(max(int(requested), 1), MAX_COMPANIES_PER_IMPACT)


def resolve_company_discovery_mode() -> str:
    raw = os.getenv("POLICYCHAIN_COMPANY_DISCOVERY_MODE", "web_first").strip().lower()
    return "legacy_cnfinancial" if raw == "legacy_cnfinancial" else "web_first"


def _build_web_first_unavailable_coverage(
    industry_impacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "impact_id": str(impact.get("impact_id") or f"IMP-{index:03d}"),
            "industry": str(impact.get("industry") or ""),
            "chain_segment": str(impact.get("chain_segment") or ""),
            "business_variables": list(impact.get("business_variables") or []),
            "affected_company_types": list(impact.get("affected_company_types") or []),
            "candidate_count": 0,
            "passed_count": 0,
            "rejected_count": 0,
            "company_names": [],
            "retrieval_status": "discovery_error",
            "coverage_status": "discovery_error",
            "no_match_reason": "Web-first discovery 未执行；为避免静默切换候选语义，未调用旧 CNFinancial-first 召回。",
        }
        for index, impact in enumerate(industry_impacts, start=1)
    ]


def _dedupe_matches(matches: list[CompanyMatch]) -> list[CompanyMatch]:
    seen: set[tuple[str, str, str]] = set()
    output: list[CompanyMatch] = []
    for match in matches:
        key = (match.impact_id, match.company_name, match.stock_code)
        if key not in seen:
            seen.add(key)
            output.append(match)
    return output


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


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value)).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _candidate_retrieval_uncertainties(retrieval_by_impact: dict[str, dict[str, Any]]) -> list[str]:
    statuses = {str(item.get("status") or "empty") for item in retrieval_by_impact.values()}
    uncertainties: list[str] = []
    if "error" in statuses:
        uncertainties.append("部分或全部 CNFinancial 候选公司查询失败，不能将查询失败解释为真实无候选。")
    if "unavailable" in statuses:
        uncertainties.append("部分或全部 CNFinancial 候选公司工具不可用或已熔断，本次未完成对应路径的候选检索。")
    if statuses and statuses <= {"empty"}:
        uncertainties.append("CNFinancial 候选查询成功但真实返回空，未形成足够 A 股公司候选。")
    if not uncertainties:
        uncertainties.append("未形成足够 CNFinancial A 股候选公司；请检查合法行业板块选择、工具状态和路径特异搜索词。")
    return uncertainties
