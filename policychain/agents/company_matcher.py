from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import re
from typing import Any

from policychain.mcp import MCPToolInvoker, consume_mcp_invoker_errors, is_unavailable_invoker
from policychain.safety import assert_no_investment_advice
from policychain.schemas.agent_outputs import CompanyEvidence, CompanyMatch, CompanyMatchOutput
from policychain.state import PolicyResearchState
from policychain.tools import collect_company_candidates, collect_company_web_evidence, read_company_source
from policychain.tools.mcp_tools import mcp_unavailable_uncertainty


MATCH_LEVELS = ("high", "medium", "low")
DEFAULT_MAX_COMPANIES_PER_IMPACT = 3

DOMAIN_TERMS = (
    "钢铁",
    "冶炼",
    "轧钢",
    "节能",
    "环保",
    "绿色",
    "低碳",
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


class CompanyMatchError(RuntimeError):
    """Raised when Company Matcher cannot produce a structured result."""


def run_company_matcher(
    state: PolicyResearchState,
    top_k_per_industry: int = DEFAULT_MAX_COMPANIES_PER_IMPACT,
    mcp_invoker: MCPToolInvoker | None = None,
) -> CompanyMatchOutput:
    """Run deterministic company-business matching from industry impacts."""

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

    companies, coverage, audit_logs = match_candidate_records_to_impacts(
        industry_impacts=industry_impacts,
        candidate_records=candidate_records,
        top_k_per_industry=top_k_per_industry,
    )
    tool_logs.extend(_audit_tool_logs(coverage))

    uncertainties = [
        "公司资料来自 CNFinancial/Web 候选证据；MCP 不可用或返回空时，不再使用本地 mock 公司填充用户报告。",
        "公司部分仅表示业务相关性研究清单，不构成任何投资建议。",
    ]
    if not candidate_records:
        uncertainties.append("未形成足够 CNFinancial A 股候选公司；请检查合法行业板块选择、CNFinancial 工具可用性和搜索关键词。")
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
) -> CompanyMatchOutput:
    """Bind LLM company matches to concrete impact paths and audit relevance."""

    candidate_by_name = _candidate_index(candidate_records)
    audited: list[CompanyMatch] = []
    audit_logs: list[dict[str, Any]] = []
    matches_by_impact: dict[str, list[CompanyMatch]] = {}

    for match in output.companies:
        record = candidate_by_name.get(_company_key(match.company_name, match.stock_code))
        if not record:
            record = candidate_by_name.get(_company_key(match.company_name, ""))
        impact_index, impact = _best_impact_for_match(match, record or {}, industry_impacts)
        status = _audit_candidate_against_impact(record or _company_match_as_record(match), impact)
        audited_match = _apply_audit(match, impact_index, impact, status)
        audit_logs.append(_audit_log_entry(audited_match, status, impact_index, impact))
        if status["decision"] == "reject":
            continue
        matches_by_impact.setdefault(_impact_id(impact_index), []).append(audited_match)

    for impact_index, _impact in enumerate(industry_impacts, start=1):
        impact_id = _impact_id(impact_index)
        scoped = sorted(matches_by_impact.get(impact_id, []), key=lambda item: item.confidence, reverse=True)
        audited.extend(scoped[:top_k_per_industry])

    coverage = _build_coverage_matrix(
        industry_impacts=industry_impacts,
        candidate_records=candidate_records,
        matches=audited,
        audit_logs=audit_logs,
    )
    audited_output = CompanyMatchOutput(companies=audited, uncertainties=list(output.uncertainties))
    setattr(audited_output, "_company_coverage", coverage)
    setattr(audited_output, "_audit_logs", audit_logs)
    return audited_output


def match_candidate_records_to_impacts(
    industry_impacts: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    top_k_per_industry: int = DEFAULT_MAX_COMPANIES_PER_IMPACT,
) -> tuple[list[CompanyMatch], list[dict[str, Any]], list[dict[str, Any]]]:
    matches: list[CompanyMatch] = []
    audit_logs: list[dict[str, Any]] = []

    for impact_index, impact in enumerate(industry_impacts, start=1):
        scoped: list[CompanyMatch] = []
        for record in candidate_records:
            status = _audit_candidate_against_impact(record, impact)
            provisional = _build_company_match(record, impact_index, impact, status)
            audit_logs.append(_audit_log_entry(provisional, status, impact_index, impact))
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
    shared_terms = _shared_terms(company_text, impact_text)
    has_business_evidence = bool(str(record.get("business_evidence") or record.get("matched_business") or "").strip())
    source_tool = str(record.get("candidate_source_tool") or "")
    confidence = _confidence(record, impact, shared_terms, has_business_evidence)
    negative_evidence: list[str] = []

    if not record.get("company_name"):
        return {
            "decision": "reject",
            "match_level": "low",
            "confidence": 0.0,
            "reason": "候选记录缺少公司名称。",
            "shared_terms": [],
            "negative_evidence": ["候选记录缺少公司名称。"],
        }

    if not shared_terms:
        negative_evidence.append("未发现公司业务与该政策路径的产业链环节或经营变量存在明确交集。")
    if not has_business_evidence:
        negative_evidence.append("缺少主营业务、产品服务、公告或官网等业务证据。")
    if source_tool in WEAK_SOURCE_TOOLS and not has_business_evidence:
        negative_evidence.append("候选仅来自关键词搜索，缺少业务证据支撑。")

    if not shared_terms and not has_business_evidence:
        return {
            "decision": "reject",
            "match_level": "low",
            "confidence": min(confidence, 0.25),
            "reason": "仅有行业或概念线索，缺少业务交集和业务证据。",
            "shared_terms": shared_terms,
            "negative_evidence": negative_evidence,
        }

    if not shared_terms:
        return {
            "decision": "reject",
            "match_level": "low",
            "confidence": min(confidence, 0.35),
            "reason": "有业务资料但未能直接对应该行业路径，剔除该路径下的公司匹配。",
            "shared_terms": shared_terms,
            "negative_evidence": negative_evidence,
        }

    if source_tool in WEAK_SOURCE_TOOLS and len(shared_terms) < 2:
        return {
            "decision": "keep_low",
            "match_level": "low",
            "confidence": min(confidence, 0.55),
            "reason": "公司来自关键词补充搜索，业务交集较弱，不能提升为中高置信。",
            "shared_terms": shared_terms,
            "negative_evidence": negative_evidence,
        }

    return {
        "decision": "passed",
        "match_level": _match_level(confidence),
        "confidence": confidence,
        "reason": f"公司业务证据与路径关键词存在交集：{', '.join(shared_terms[:5])}。",
        "shared_terms": shared_terms,
        "negative_evidence": negative_evidence,
    }


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
    return round(min(score, 0.92), 2)


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
) -> list[dict[str, Any]]:
    coverage: list[dict[str, Any]] = []
    for index, impact in enumerate(industry_impacts, start=1):
        impact_id = _impact_id(index)
        scoped_matches = [match for match in matches if match.impact_id == impact_id]
        scoped_audits = [item for item in audit_logs if item.get("impact_id") == impact_id]
        rejected_count = sum(1 for item in scoped_audits if item.get("decision") == "reject")
        no_match_reason = ""
        if not candidate_records:
            no_match_reason = "CNFinancial 未返回可用于该路径的 A 股候选公司。"
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
                "candidate_count": len(candidate_records),
                "passed_count": len(scoped_matches),
                "rejected_count": rejected_count,
                "company_names": [match.company_name for match in scoped_matches],
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
    return {
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
        "reason": status["reason"],
    }


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
        status = _audit_candidate_against_impact(candidate_record, impact)
        score = float(status["confidence"])
        if status["decision"] == "reject":
            score -= 0.2
        if score > best_score:
            best_score = score
            best_index = index
            best_impact = impact
    return best_index, best_impact


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
