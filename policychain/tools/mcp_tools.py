from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import os
import re
from time import perf_counter
from typing import Any
from uuid import uuid4

from policychain.mcp import (
    MCPToolError,
    MCPToolInvoker,
    MCPToolUnavailable,
    UnavailableMCPInvoker,
    mcp_payload_error_message,
)


OPEN_WEBSEARCH_SERVER = "web-search"
CNFINANCIAL_SERVER = "cn-financial"

OPEN_WEBSEARCH_SEARCH_TOOL = "search"
OPEN_WEBSEARCH_FETCH_TOOL = "fetchWebContent"

CNFINANCIAL_IMPACT_TOOLS = (
    "get_industry_list",
    "get_concept_list",
    "get_industry_stocks",
    "get_sector_fund_flow",
    "get_industry_pe",
    "search_news",
    "get_macro_gdp",
    "get_macro_cpi",
    "get_macro_pmi",
    "get_macro_money_supply",
)

CNFINANCIAL_COMPANY_TOOLS = (
    "search_stock",
    "get_company_info",
    "get_company_profile",
    "get_segments_revenue",
    "get_financial_indicators",
    "get_growth_rates",
    "get_competitors",
    "get_company_announcements",
    "get_stock_news",
)

DEFAULT_COMPANY_ENRICHMENT_TOOLS = (
    "get_company_profile",
)

CNFINANCIAL_INDUSTRY_TOOLS_WITH_TERM = {"get_industry_stocks", "get_industry_pe"}
CNFINANCIAL_LIST_TOOLS = {"get_industry_list", "get_concept_list", "get_sector_fund_flow"}
CNFINANCIAL_MACRO_TOOLS = {"get_macro_gdp", "get_macro_cpi", "get_macro_pmi", "get_macro_money_supply"}

DEFAULT_COMPANY_INDUSTRY_BOARDS = (
    "\u8f6f\u4ef6\u5f00\u53d1",  # 软件开发
    "\u4e92\u8054\u7f51\u670d\u52a1",  # 互联网服务
    "\u8ba1\u7b97\u673a\u8bbe\u5907",  # 计算机设备
)

INDUSTRY_BOARD_MAPPINGS = (
    (
        (
            "\u751f\u6210\u5f0f\u4eba\u5de5\u667a\u80fd",
            "\u4eba\u5de5\u667a\u80fd",
            "\u7b97\u6cd5",
            "\u6a21\u578b",
            "\u8bad\u7ec3\u6570\u636e",
        ),
        (
            "\u8f6f\u4ef6\u5f00\u53d1",
            "\u4e92\u8054\u7f51\u670d\u52a1",
            "\u8ba1\u7b97\u673a\u8bbe\u5907",
        ),
    ),
    (
        ("\u6570\u636e", "\u5b89\u5168", "\u5408\u89c4", "\u8bc4\u4f30", "\u5907\u6848"),
        (
            "\u8f6f\u4ef6\u5f00\u53d1",
            "IT\u670d\u52a1",
            "\u4e92\u8054\u7f51\u670d\u52a1",
        ),
    ),
    (
        ("\u5185\u5bb9", "\u6807\u8bc6", "\u672a\u6210\u5e74\u4eba", "\u4f20\u64ad"),
        (
            "\u6587\u5316\u4f20\u5a92",
            "\u4e92\u8054\u7f51\u670d\u52a1",
            "\u8f6f\u4ef6\u5f00\u53d1",
        ),
    ),
)

COMPANY_NAME_KEYS = (
    "company_name",
    "name",
    "stock_name",
    "\u540d\u79f0",
    "\u80a1\u7968\u7b80\u79f0",
    "\u8bc1\u5238\u7b80\u79f0",
)
COMPANY_CODE_KEYS = (
    "stock_code",
    "code",
    "symbol",
    "\u4ee3\u7801",
    "\u80a1\u7968\u4ee3\u7801",
    "\u8bc1\u5238\u4ee3\u7801",
)
BUSINESS_KEYS = (
    "matched_business",
    "main_business",
    "business",
    "description",
    "summary",
    "\u4e3b\u8425\u4e1a\u52a1",
    "\u4e1a\u52a1\u8303\u56f4",
)
REVENUE_KEYS = (
    "revenue_relevance",
    "revenue_ratio",
    "ratio",
    "\u6536\u5165\u5360\u6bd4",
    "\u8425\u6536\u5360\u6bd4",
    "\u4e3b\u8425\u5360\u6bd4",
)
SECTOR_NAME_KEYS = (
    "name",
    "title",
    "industry",
    "concept",
    "sector",
    "board_name",
    "\u540d\u79f0",
    "\u677f\u5757\u540d\u79f0",
    "\u884c\u4e1a",
    "\u884c\u4e1a\u540d\u79f0",
    "\u6982\u5ff5",
    "\u6982\u5ff5\u540d\u79f0",
)

POLICY_WEB_TOPICS = (
    "历史政策",
    "上位政策",
    "官方解读",
    "实施细则",
    "地方配套政策",
    "资金安排",
    "试点名单",
    "后续执行情况",
)

POLICY_SOURCE_PRIORITY = ("政府官网", "主管部门", "官方解读")
INDUSTRY_SOURCE_PRIORITY = ("政府部门", "国家统计局", "行业协会", "权威机构")
COMPANY_SOURCE_PRIORITY = ("交易所公告", "巨潮资讯", "公司官网")

COMPANY_BUSINESS_EVIDENCE_KEYWORDS = (
    "产品",
    "服务",
    "主营业务",
    "分部收入",
    "收入占比",
    "销量",
    "产量",
    "产能",
    "产能利用率",
    "在建项目",
    "扩产",
    "客户",
    "订单",
    "研发",
    "补贴",
    "资质",
    "政策风险",
)


def search_web(
    query: str,
    source_priority: Iterable[str] | None = None,
    top_k: int = 5,
    invoker: MCPToolInvoker | None = None,
    tool_logs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not query.strip():
        return []

    raw = _invoke_or_empty(
        invoker=invoker,
        server_name=OPEN_WEBSEARCH_SERVER,
        tool_name=OPEN_WEBSEARCH_SEARCH_TOOL,
        arguments={"query": query, "limit": top_k},
        tool_logs=tool_logs,
    )
    return normalize_mcp_evidence(
        raw,
        query=query,
        server_name=OPEN_WEBSEARCH_SERVER,
        tool_name=OPEN_WEBSEARCH_SEARCH_TOOL,
        source_priority=source_priority,
    )[:top_k]


def fetch_web_content(
    url: str,
    max_chars: int = 30000,
    invoker: MCPToolInvoker | None = None,
    tool_logs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not url.strip():
        return []
    raw = _invoke_or_empty(
        invoker=invoker,
        server_name=OPEN_WEBSEARCH_SERVER,
        tool_name=OPEN_WEBSEARCH_FETCH_TOOL,
        arguments={"url": url, "maxChars": max_chars},
        tool_logs=tool_logs,
    )
    return normalize_mcp_evidence(
        raw,
        query=url,
        server_name=OPEN_WEBSEARCH_SERVER,
        tool_name=OPEN_WEBSEARCH_FETCH_TOOL,
    )


def collect_policy_web_evidence(
    user_query: str,
    invoker: MCPToolInvoker | None = None,
    top_k_per_topic: int = 2,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    max_topics = _mcp_int_setting(
        "POLICYCHAIN_MCP_MAX_POLICY_WEB_TOPICS",
        1 if _mcp_fast_mode() else len(POLICY_WEB_TOPICS),
        minimum=0,
    )
    for topic in list(POLICY_WEB_TOPICS)[:max_topics]:
        evidence.extend(
            search_web(
                query=f"{user_query} {topic}",
                source_priority=POLICY_SOURCE_PRIORITY,
                top_k=top_k_per_topic,
                invoker=invoker,
            )
        )
    return _dedupe_evidence(evidence)


def collect_impact_research(
    policy_analysis: dict[str, Any],
    industry_impacts: list[dict[str, Any]],
    invoker: MCPToolInvoker | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    financial: list[dict[str, Any]] = []
    web: list[dict[str, Any]] = []
    tool_logs: list[dict[str, Any]] = []
    sector_selections: list[dict[str, Any]] = []

    industry_catalog, concept_catalog = _load_cnfinancial_sector_catalogs(invoker, tool_logs)
    for log in tool_logs[-2:]:
        log["internal_only"] = True

    fast_mode = _mcp_fast_mode()
    if not fast_mode:
        for tool_name in CNFINANCIAL_MACRO_TOOLS:
            raw = _invoke_or_empty(
                invoker=invoker,
                server_name=CNFINANCIAL_SERVER,
                tool_name=tool_name,
                arguments={},
                tool_logs=tool_logs,
            )
            financial.extend(
                normalize_mcp_evidence(
                    raw,
                    query=tool_name,
                    server_name=CNFINANCIAL_SERVER,
                    tool_name=tool_name,
                )
            )

        for sector_type in ("\u884c\u4e1a\u8d44\u91d1\u6d41", "\u6982\u5ff5\u8d44\u91d1\u6d41"):
            raw = _invoke_or_empty(
                invoker=invoker,
                server_name=CNFINANCIAL_SERVER,
                tool_name="get_sector_fund_flow",
                arguments={"sector_type": sector_type, "indicator": "\u4eca\u65e5"},
                tool_logs=tool_logs,
            )
            financial.extend(
                normalize_mcp_evidence(
                    raw,
                    query=sector_type,
                    server_name=CNFINANCIAL_SERVER,
                    tool_name="get_sector_fund_flow",
                )
            )

    scoped_impacts = industry_impacts or [{"industry": " ".join(_industry_terms(policy_analysis, []))}]
    for impact_index, impact in enumerate(scoped_impacts, start=1):
        impact_id = _impact_identifier(impact, impact_index)
        impact_terms = _industry_terms({}, [impact]) or _industry_terms(policy_analysis, [])
        selection = _select_cnfinancial_sectors(
            terms=impact_terms,
            industry_impacts=[impact],
            industry_catalog=industry_catalog,
            concept_catalog=concept_catalog,
            max_industries=_mcp_int_setting("POLICYCHAIN_MCP_MAX_SELECTED_INDUSTRIES", 2, minimum=1),
            max_concepts=_mcp_int_setting("POLICYCHAIN_MCP_MAX_SELECTED_CONCEPTS", 3, minimum=0),
        )
        selection["impact_id"] = impact_id
        sector_selections.append(selection)
        _append_sector_selection_log(tool_logs, selection, impact_id=impact_id)
        financial.append(_sector_selection_evidence(selection))

        for industry in _selected_sector_names(selection, "selected_industries"):
            industry_tools = ("get_industry_stocks",) if fast_mode else ("get_industry_stocks", "get_industry_pe")
            for tool_name in industry_tools:
                raw = _invoke_or_empty(
                    invoker=invoker,
                    server_name=CNFINANCIAL_SERVER,
                    tool_name=tool_name,
                    arguments={"industry": industry},
                    tool_logs=tool_logs,
                    log_context={"impact_id": impact_id, "sector": industry},
                )
                financial.extend(
                    _tag_evidence(
                        normalize_mcp_evidence(
                            raw,
                            query=industry,
                            server_name=CNFINANCIAL_SERVER,
                            tool_name=tool_name,
                        ),
                        impact_id=impact_id,
                        sector=industry,
                    )
                )

        if not fast_mode:
            search_terms = _limited(
                _sector_search_terms(selection, impact_terms),
                _mcp_int_setting("POLICYCHAIN_MCP_MAX_SEARCH_TERMS", 2, minimum=1),
            )
            for term in search_terms:
                raw = _invoke_or_empty(
                    invoker=invoker,
                    server_name=CNFINANCIAL_SERVER,
                    tool_name="search_news",
                    arguments={"keyword": term, "num_results": 5},
                    tool_logs=tool_logs,
                    log_context={"impact_id": impact_id, "keyword": term},
                )
                financial.extend(
                    _tag_evidence(
                        normalize_mcp_evidence(
                            raw,
                            query=term,
                            server_name=CNFINANCIAL_SERVER,
                            tool_name="search_news",
                        ),
                        impact_id=impact_id,
                        keyword=term,
                    )
                )

            for term in _limited(impact_terms, _mcp_int_setting("POLICYCHAIN_MCP_MAX_SEARCH_TERMS", 2, minimum=1)):
                web.extend(
                    _tag_evidence(
                        search_web(
                            query=f"{term} industry data statistics association production sales price inventory capacity technology route",
                            source_priority=INDUSTRY_SOURCE_PRIORITY,
                            top_k=top_k,
                            invoker=invoker,
                            tool_logs=tool_logs,
                        ),
                        impact_id=impact_id,
                        keyword=term,
                    )
                )

    return {
        "cnfinancial": _dedupe_evidence(financial),
        "web": _dedupe_evidence(web),
        "tool_logs": tool_logs,
        "sector_selection": sector_selections[0] if sector_selections else {},
        "sector_selections": sector_selections,
    }


def collect_company_candidates(
    industry_impacts: list[dict[str, Any]],
    invoker: MCPToolInvoker | None = None,
    top_k_per_industry: int = 3,
    tool_logs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if top_k_per_industry <= 0:
        raise ValueError("top_k_per_industry must be positive")
    max_candidates_per_impact = _mcp_int_setting("POLICYCHAIN_MCP_MAX_COMPANY_CANDIDATES", 5, minimum=1)
    industry_catalog, concept_catalog = _load_cnfinancial_sector_catalogs(invoker, tool_logs)
    if tool_logs is not None:
        for log in tool_logs[-2:]:
            log["internal_only"] = True

    path_candidates: list[dict[str, Any]] = []
    for impact_index, impact in enumerate(industry_impacts, start=1):
        impact_id = _impact_identifier(impact, impact_index)
        impact_terms = _industry_terms({}, [impact])
        selection = _select_cnfinancial_sectors(
            terms=impact_terms,
            industry_impacts=[impact],
            industry_catalog=industry_catalog,
            concept_catalog=concept_catalog,
            max_industries=_mcp_int_setting("POLICYCHAIN_MCP_MAX_SELECTED_INDUSTRIES", 2, minimum=1),
            max_concepts=_mcp_int_setting("POLICYCHAIN_MCP_MAX_SELECTED_CONCEPTS", 3, minimum=0),
        )
        selection["impact_id"] = impact_id
        _append_sector_selection_log(tool_logs, selection, impact_id=impact_id)
        legal_industries = _selected_sector_names(selection, "selected_industries")
        selected_concepts = _selected_sector_names(selection, "selected_concepts")
        recalled: list[dict[str, Any]] = []

        for industry in legal_industries:
            raw, call_log = _invoke_with_log(
                invoker=invoker,
                server_name=CNFINANCIAL_SERVER,
                tool_name="get_industry_stocks",
                arguments={"industry": industry},
                tool_logs=tool_logs,
                log_context={"impact_id": impact_id, "sector": industry, "source_type": "raw_recall"},
            )
            items = _payload_items(raw)
            for item in items[:max_candidates_per_impact]:
                candidate = _candidate_from_recall(
                    item,
                    impact,
                    impact_id=impact_id,
                    tool_name="get_industry_stocks",
                    tool_call_id=str(call_log["tool_call_id"]),
                    sector=industry,
                )
                if not _is_valid_a_share_candidate(candidate):
                    continue
                candidate["selected_industries"] = legal_industries
                candidate["selected_concepts"] = selected_concepts
                recalled.append(candidate)
            call_log["raw_count"] = len(items)
            call_log["normalized_count"] = len(recalled)
            call_log["truncated"] = len(items) > max_candidates_per_impact

        search_terms = _unique(
            [
                *selected_concepts,
                *_limited(
                    _sector_search_terms(selection, impact_terms),
                    _mcp_int_setting("POLICYCHAIN_MCP_MAX_SEARCH_TERMS", 2, minimum=1),
                ),
                *_candidate_stock_search_terms(impact)[:1],
            ]
        )
        if not search_terms:
            search_terms = [_impact_candidate_term(impact) or "A股"]
        for term in search_terms:
            raw, call_log = _invoke_with_log(
                invoker=invoker,
                server_name=CNFINANCIAL_SERVER,
                tool_name="search_stock",
                arguments={"keyword": term},
                tool_logs=tool_logs,
                log_context={"impact_id": impact_id, "keyword": term, "source_type": "raw_recall"},
            )
            items = _payload_items(raw)
            accepted_before = len(recalled)
            for item in items[:max_candidates_per_impact]:
                candidate = _candidate_from_recall(
                    item,
                    impact,
                    impact_id=impact_id,
                    tool_name="search_stock",
                    tool_call_id=str(call_log["tool_call_id"]),
                    keyword=term,
                )
                if not _is_valid_a_share_candidate(candidate):
                    continue
                candidate["selected_industries"] = legal_industries
                candidate["selected_concepts"] = selected_concepts
                recalled.append(candidate)
            call_log["raw_count"] = len(items)
            call_log["normalized_count"] = len(recalled) - accepted_before
            call_log["truncated"] = len(items) > max_candidates_per_impact

        deduped = _dedupe_and_merge_companies(recalled)
        kept = deduped[:max_candidates_per_impact]
        _append_candidate_summary_log(
            tool_logs,
            impact_id=impact_id,
            raw_count=len(recalled),
            dedup_count=len(deduped),
            kept_count=len(kept),
        )
        path_candidates.extend(kept)

    candidates = _dedupe_and_merge_companies(path_candidates)
    for candidate in candidates:
        _enrich_company_candidate(candidate, invoker=invoker, tool_logs=tool_logs)
        candidate["enriched"] = True
    return candidates


def merge_react_company_candidates(
    company_records: list[dict[str, Any]],
    industry_impacts: list[dict[str, Any]],
    react_evidence: list[dict[str, Any]],
    invoker: MCPToolInvoker | None = None,
    tool_logs: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Promote only validated CNFinancial ReAct results into the candidate pool."""

    impacts = {
        _impact_identifier(impact, index): impact
        for index, impact in enumerate(industry_impacts, start=1)
    }
    max_candidates_per_impact = _mcp_int_setting("POLICYCHAIN_MCP_MAX_COMPANY_CANDIDATES", 5, minimum=1)
    existing_keys = {_company_identity(record) for record in company_records}
    path_keys: dict[str, set[tuple[str, str]]] = {impact_id: set() for impact_id in impacts}
    for record in company_records:
        for impact_id in record.get("impact_ids") or _provenance_impact_ids(record):
            path_keys.setdefault(str(impact_id), set()).add(_company_identity(record))

    promoted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for evidence in react_evidence:
        impact_id = str(evidence.get("impact_id") or "")
        server_name = str(evidence.get("server_name") or "")
        tool_name = str(evidence.get("tool_name") or "")
        base = {
            "time": datetime.now(timezone.utc).isoformat(),
            "impact_id": impact_id,
            "company_name": str(evidence.get("title") or evidence.get("company_name") or ""),
            "stock_code": str(evidence.get("stock_code") or ""),
            "tool": tool_name,
            "tool_call_id": str(evidence.get("tool_call_id") or ""),
            "react_step": evidence.get("react_step"),
            "source_type": str(evidence.get("source_type") or ""),
        }
        if server_name != CNFINANCIAL_SERVER or tool_name not in {"search_stock", "get_industry_stocks"}:
            audit.append({**base, "decision": "reject", "reason": "仅合法 CNFinancial 公司工具结果可进入候选白名单；Web 结果只作为业务证据。"})
            continue
        impact = impacts.get(impact_id)
        if impact is None:
            audit.append({**base, "decision": "reject", "reason": "ReAct 候选缺少合法 impact_id 路径绑定。"})
            continue
        raw_item = evidence.get("raw_payload")
        if not isinstance(raw_item, dict):
            audit.append({**base, "decision": "reject", "reason": "CNFinancial ReAct 结果缺少可验证的公司原始字段。"})
            continue
        candidate = _candidate_from_recall(
            raw_item,
            impact,
            impact_id=impact_id,
            tool_name=tool_name,
            tool_call_id=str(evidence.get("tool_call_id") or f"react-{uuid4().hex}"),
            sector=str((evidence.get("raw_payload") or {}).get("industry") or "") if tool_name == "get_industry_stocks" else "",
            keyword=str(evidence.get("query") or "") if tool_name == "search_stock" else "",
            react_step=evidence.get("react_step"),
            source_type="cnfinancial_react",
        )
        base.update(
            {
                "company_name": str(candidate.get("company_name") or ""),
                "stock_code": str(candidate.get("stock_code") or ""),
            }
        )
        if not _is_valid_a_share_candidate(candidate):
            audit.append({**base, "decision": "reject", "reason": "ReAct 候选缺少有效公司名称或 6 位 A 股代码。"})
            continue
        identity = _company_identity(candidate)
        if identity not in path_keys.setdefault(impact_id, set()) and len(path_keys[impact_id]) >= max_candidates_per_impact:
            audit.append({**base, "decision": "reject", "reason": "该路径候选池已达到配置上限。"})
            continue
        path_keys[impact_id].add(identity)
        promoted.append(candidate)
        audit.append({**base, "decision": "accept", "reason": "合法 CNFinancial ReAct 公司结果已规范化并绑定到路径。"})

    merged = _dedupe_and_merge_companies([*company_records, *promoted])
    for candidate in merged:
        if _company_identity(candidate) not in existing_keys:
            _enrich_company_candidate(candidate, invoker=invoker, tool_logs=tool_logs)
            candidate["enriched"] = True
    return merged, audit


def _provenance_impact_ids(record: dict[str, Any]) -> list[str]:
    return _unique(
        [str(item.get("impact_id") or "") for item in record.get("provenance") or [] if isinstance(item, dict)]
    )


def collect_company_web_evidence(
    company_records: list[dict[str, Any]],
    invoker: MCPToolInvoker | None = None,
    top_k: int = 2,
    tool_logs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if _mcp_fast_mode():
        _append_fast_mode_skip_log(tool_logs, "company_web_evidence")
        return []
    evidence: list[dict[str, Any]] = []
    for company in company_records:
        name = str(company.get("company_name") or "")
        code = str(company.get("stock_code") or "")
        if not name:
            continue
        evidence.extend(
            search_web(
                query=f"{name} {code} 公司官网 交易所公告 巨潮资讯 主营业务 公告",
                source_priority=COMPANY_SOURCE_PRIORITY,
                top_k=top_k,
                invoker=invoker,
                tool_logs=tool_logs,
            )
        )
    return _dedupe_evidence(evidence)


def normalize_mcp_evidence(
    raw_payload: Any,
    query: str,
    server_name: str,
    tool_name: str,
    source_priority: Iterable[str] | None = None,
    query_time: str | None = None,
) -> list[dict[str, Any]]:
    query_time = query_time or datetime.now(timezone.utc).isoformat()
    evidence: list[dict[str, Any]] = []
    for item in _payload_items(raw_payload):
        title = _first_value(
            item,
            (
                "title",
                "name",
                "report_title",
                "announcement_title",
                "announcementTitle",
                "stock_name",
                "\u540d\u79f0",
                "\u80a1\u7968\u7b80\u79f0",
                "\u8bc1\u5238\u7b80\u79f0",
            ),
        )
        source_url = _first_value(item, ("url", "link", "source_url", "original_source_url", "pdf_url", "download_url", "adjunctUrl"))
        source_org = _first_value(item, ("source_org", "source", "agency", "publisher", "org", "engine", "secName"))
        published_date = _first_value(item, ("publish_date", "published_date", "date", "year", "announcement_time", "announcementTime"))
        summary = _first_value(item, ("summary", "description", "snippet", "content", "text", "business_evidence", *BUSINESS_KEYS))
        if not any((title, source_url, source_org, published_date, summary)):
            continue
        evidence.append(
            {
                "title": str(title or ""),
                "source_org": str(source_org or ""),
                "published_date": str(published_date or ""),
                "source_url": str(source_url or ""),
                "summary": _clip(str(summary or "")),
                "query": query,
                "query_time": query_time,
                "server_name": server_name,
                "tool_name": tool_name,
                "source_priority": list(source_priority or []),
                "raw_payload": item,
            }
        )
    return evidence


def mcp_unavailable_uncertainty(tool_family: str) -> str:
    return f"{tool_family} MCP 未配置或不可用，本次仅使用已可用的本地/离线证据。"


def _invoke_or_empty(
    invoker: MCPToolInvoker | None,
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any],
    tool_logs: list[dict[str, Any]] | None = None,
    log_context: dict[str, Any] | None = None,
) -> Any:
    raw, _log_entry = _invoke_with_log(
        invoker=invoker,
        server_name=server_name,
        tool_name=tool_name,
        arguments=arguments,
        tool_logs=tool_logs,
        log_context=log_context,
    )
    return raw


def _invoke_with_log(
    invoker: MCPToolInvoker | None,
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any],
    tool_logs: list[dict[str, Any]] | None = None,
    log_context: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    active_invoker = invoker or UnavailableMCPInvoker()
    log_entry = _new_tool_log(server_name=server_name, tool_name=tool_name, arguments=arguments)
    log_entry.update(log_context or {})
    started = perf_counter()
    try:
        raw = active_invoker.invoke(server_name, tool_name, arguments)
        error_message = mcp_payload_error_message(raw, server_name=server_name, tool_name=tool_name)
        if error_message:
            log_entry.update({"status": "error", "error": error_message, "count": 0, "duration_ms": _elapsed_ms(started)})
            _append_tool_log(tool_logs, log_entry)
            _record_invoker_error(active_invoker, error_message)
            _record_mcp_event(log_entry)
            return [], log_entry
        log_entry.update({"status": "ok", "count": _payload_count(raw), "error": "", "duration_ms": _elapsed_ms(started)})
        _append_tool_log(tool_logs, log_entry)
        _record_mcp_event(log_entry)
        return raw, log_entry
    except (MCPToolError, MCPToolUnavailable) as exc:
        log_entry.update({"status": "error", "error": str(exc), "count": 0, "duration_ms": _elapsed_ms(started)})
        _append_tool_log(tool_logs, log_entry)
        _record_invoker_error(active_invoker, str(exc))
        _record_mcp_event(log_entry)
        return [], log_entry


def _new_tool_log(server_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "tool_call_id": f"tool-{uuid4().hex}",
        "server_name": server_name,
        "tool_name": tool_name,
        "arguments": dict(arguments),
        "status": "pending",
        "count": 0,
        "error": "",
    }


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def _record_mcp_event(log_entry: dict[str, Any]) -> None:
    try:
        from policychain.observability import record_event

        record_event("mcp.call", **log_entry)
    except Exception:
        return


def _append_tool_log(tool_logs: list[dict[str, Any]] | None, log_entry: dict[str, Any]) -> None:
    if tool_logs is not None:
        tool_logs.append(log_entry)


def _payload_count(payload: Any) -> int:
    return len(_payload_items(payload))


def _record_invoker_error(invoker: MCPToolInvoker, message: str) -> None:
    errors = getattr(invoker, "errors", None)
    if isinstance(errors, list) and message not in errors:
        errors.append(message)
        return
    inner = getattr(invoker, "inner", None)
    inner_errors = getattr(inner, "errors", None)
    if isinstance(inner_errors, list) and message not in inner_errors:
        inner_errors.append(message)


def _payload_items(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("results", "result", "data", "items", "reports", "announcements", "stocks", "companies"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                return [value]
        return [payload]
    return []


def _first_value(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _load_cnfinancial_sector_catalogs(
    invoker: MCPToolInvoker | None,
    tool_logs: list[dict[str, Any]] | None = None,
) -> tuple[Any, Any]:
    industry_catalog = _invoke_or_empty(
        invoker=invoker,
        server_name=CNFINANCIAL_SERVER,
        tool_name="get_industry_list",
        arguments={},
        tool_logs=tool_logs,
    )
    concept_catalog = _invoke_or_empty(
        invoker=invoker,
        server_name=CNFINANCIAL_SERVER,
        tool_name="get_concept_list",
        arguments={},
        tool_logs=tool_logs,
    )
    return industry_catalog, concept_catalog


def _select_cnfinancial_sectors(
    terms: list[str],
    industry_impacts: list[dict[str, Any]],
    industry_catalog: Any,
    concept_catalog: Any,
    max_industries: int = 4,
    max_concepts: int = 4,
) -> dict[str, Any]:
    keyword_terms = _sector_keyword_terms(terms, industry_impacts)
    industry_names = _extract_sector_names(industry_catalog)
    concept_names = _extract_sector_names(concept_catalog)
    selected_industries = _rank_sector_names(industry_names, keyword_terms, max_items=max_industries)
    selected_concepts = _rank_sector_names(concept_names, keyword_terms, max_items=max_concepts)
    return {
        "keywords": keyword_terms,
        "selected_industries": selected_industries,
        "selected_concepts": selected_concepts,
        "industry_catalog_available": bool(industry_names),
        "concept_catalog_available": bool(concept_names),
    }


def _extract_sector_names(raw_catalog: Any) -> list[str]:
    names: list[str] = []
    for item in _payload_items(raw_catalog):
        value = _first_value(item, SECTOR_NAME_KEYS)
        if value:
            names.append(str(value).strip())
    return _unique([name for name in names if name])


def _sector_keyword_terms(terms: list[str], industry_impacts: list[dict[str, Any]]) -> list[str]:
    output: list[str] = []
    output.extend(terms)
    for impact in industry_impacts:
        output.extend(
            str(impact.get(key) or "")
            for key in ("industry", "chain_segment", "policy_measure", "implementation_action", "transmission_logic")
        )
        for key in ("business_variables", "affected_company_types", "conditions"):
            output.extend(str(value) for value in impact.get(key) or [])
        output.extend(_candidate_industry_terms(impact))
        output.extend(_candidate_stock_search_terms(impact))
    return _unique([_clip(term.strip(), max_chars=40) for term in output if term and term.strip()])[:20]


def _rank_sector_names(names: list[str], keyword_terms: list[str], max_items: int) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for name in names:
        score, matched_terms = _sector_match_score(name, keyword_terms)
        if score <= 0:
            continue
        ranked.append({"name": name, "score": score, "matched_terms": matched_terms})
    ranked.sort(key=lambda item: (-int(item["score"]), str(item["name"])))
    return ranked[:max_items]


def _sector_match_score(name: str, keyword_terms: list[str]) -> tuple[int, list[str]]:
    score = 0
    matched_terms: list[str] = []
    compact_name = _compact_match_text(name)
    for term in keyword_terms:
        compact_term = _compact_match_text(term)
        if not compact_term:
            continue
        if compact_term == compact_name:
            score += 20
            matched_terms.append(term)
        elif compact_term in compact_name or compact_name in compact_term:
            score += 10
            matched_terms.append(term)
        else:
            token_hits = sum(1 for token in _match_tokens(compact_term) if token and token in compact_name)
            if token_hits:
                score += min(token_hits * 2, 6)
                matched_terms.append(term)
    return score, _unique(matched_terms)[:5]


def _compact_match_text(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _match_tokens(value: str) -> list[str]:
    tokens = re.split(r"[\s,，、/;；|()（）]+", value)
    return [token for token in tokens if len(token) >= 2]


def _append_sector_selection_log(
    tool_logs: list[dict[str, Any]] | None,
    selection: dict[str, Any],
    impact_id: str | None = None,
) -> None:
    log_entry = _new_tool_log(
        server_name=CNFINANCIAL_SERVER,
        tool_name="select_sectors",
        arguments={"keywords": selection.get("keywords") or []},
    )
    selected_count = len(selection.get("selected_industries") or []) + len(selection.get("selected_concepts") or [])
    log_entry.update(
        {
            "impact_id": impact_id or selection.get("impact_id") or "",
            "status": "ok" if selected_count else "empty",
            "count": selected_count,
            "error": "",
            "selected_industries": selection.get("selected_industries") or [],
            "selected_concepts": selection.get("selected_concepts") or [],
        }
    )
    _append_tool_log(tool_logs, log_entry)


def _sector_selection_evidence(selection: dict[str, Any]) -> dict[str, Any]:
    industries = ", ".join(_selected_sector_names(selection, "selected_industries")) or "none"
    concepts = ", ".join(_selected_sector_names(selection, "selected_concepts")) or "none"
    return {
        "title": "CNFinancial sector selection",
        "source_org": "CNFinancial MCP",
        "published_date": "",
        "source_url": "",
        "summary": f"selected industries: {industries}; selected concepts: {concepts}",
        "query": "select_sectors",
        "query_time": datetime.now(timezone.utc).isoformat(),
        "server_name": CNFINANCIAL_SERVER,
        "tool_name": "select_sectors",
        "impact_id": str(selection.get("impact_id") or ""),
        "source_priority": [],
        "raw_payload": selection,
    }


def _selected_sector_names(selection: dict[str, Any], key: str) -> list[str]:
    return [str(item.get("name") or "") for item in selection.get(key) or [] if item.get("name")]


def _tag_evidence(items: list[dict[str, Any]], **context: Any) -> list[dict[str, Any]]:
    return [{**item, **{key: value for key, value in context.items() if value not in (None, "")}} for item in items]


def _impact_identifier(impact: dict[str, Any], index: int) -> str:
    return str(impact.get("impact_id") or f"IMP-{index:03d}")


def _limited(values: list[str], max_items: int) -> list[str]:
    if max_items < 0:
        return values
    return values[:max_items]


def _sector_search_terms(selection: dict[str, Any], fallback_terms: list[str]) -> list[str]:
    terms = [
        *_selected_sector_names(selection, "selected_concepts"),
        *_selected_sector_names(selection, "selected_industries"),
        *list(selection.get("keywords") or []),
        *fallback_terms,
    ]
    return _unique([_clip(str(term), max_chars=30) for term in terms if str(term).strip()])[:8]


def _industry_terms(policy_analysis: dict[str, Any], industry_impacts: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for impact in industry_impacts:
        terms.append(str(impact.get("chain_segment") or impact.get("industry") or ""))
    for key in ("target_entities", "policy_measures", "policy_goals"):
        for value in policy_analysis.get(key) or []:
            terms.append(str(value))
    return _unique([_clip(term, max_chars=40) for term in terms if term])[:3]


def _cnfinancial_impact_arguments(tool_name: str, term: str) -> dict[str, Any]:
    if tool_name in CNFINANCIAL_MACRO_TOOLS or tool_name in CNFINANCIAL_LIST_TOOLS:
        return {}
    if tool_name == "search_news":
        return {"keyword": term, "num_results": 5}
    if tool_name in CNFINANCIAL_INDUSTRY_TOOLS_WITH_TERM:
        return {"industry": term}
    return {"keyword": term}


def _cnfinancial_impact_query_terms(tool_name: str, term: str) -> list[str]:
    if tool_name in CNFINANCIAL_MACRO_TOOLS or tool_name in CNFINANCIAL_LIST_TOOLS:
        return [""]
    if tool_name in CNFINANCIAL_INDUSTRY_TOOLS_WITH_TERM:
        return _candidate_industry_terms({"industry": term, "chain_segment": term})[:2]
    return [term]


def _candidate_industry_terms(impact: dict[str, Any]) -> list[str]:
    text = _impact_text(impact)
    terms: list[str] = []
    for needles, boards in INDUSTRY_BOARD_MAPPINGS:
        if any(needle in text for needle in needles):
            terms.extend(boards)
    raw_industry = str(impact.get("industry") or impact.get("chain_segment") or "").strip()
    if raw_industry and len(raw_industry) <= 12:
        terms.append(raw_industry)
    if not terms:
        terms.extend(DEFAULT_COMPANY_INDUSTRY_BOARDS)
    return _unique(terms)[:4]


def _candidate_stock_search_terms(impact: dict[str, Any]) -> list[str]:
    text = _impact_text(impact)
    terms = [
        str(impact.get("chain_segment") or ""),
        str(impact.get("industry") or ""),
    ]
    for keyword in (
        "\u4eba\u5de5\u667a\u80fd",
        "\u7b97\u6cd5",
        "\u6a21\u578b",
        "\u6570\u636e",
        "\u5b89\u5168",
        "\u5408\u89c4",
        "\u5185\u5bb9",
    ):
        if keyword in text:
            terms.append(keyword)
    if not any(term.strip() for term in terms):
        terms.append("\u4eba\u5de5\u667a\u80fd")
    return _unique([_clip(term, max_chars=24) for term in terms if term])[:4]


def _impact_text(impact: dict[str, Any]) -> str:
    values = [
        str(impact.get("industry") or ""),
        str(impact.get("chain_segment") or ""),
        str(impact.get("transmission_logic") or ""),
        " ".join(impact.get("business_variables") or []),
        " ".join(impact.get("affected_company_types") or []),
        " ".join(impact.get("conditions") or []),
    ]
    return " ".join(values)


def _impact_candidate_term(impact: dict[str, Any]) -> str:
    return str(impact.get("chain_segment") or impact.get("industry") or impact.get("affected_company_types") or "")


def _mcp_int_setting(name: str, default: int, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return max(default, minimum)
    try:
        return max(int(raw), minimum)
    except ValueError:
        return max(default, minimum)


def _mcp_bool_setting(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _mcp_fast_mode() -> bool:
    return _mcp_bool_setting("POLICYCHAIN_MCP_FAST_MODE")


def _append_fast_mode_skip_log(tool_logs: list[dict[str, Any]] | None, tool_name: str) -> None:
    log_entry = _new_tool_log(
        server_name="policychain",
        tool_name=tool_name,
        arguments={"fast_mode": True},
    )
    log_entry.update(
        {
            "status": "skipped",
            "count": 0,
            "error": "Skipped by POLICYCHAIN_MCP_FAST_MODE",
        }
    )
    _append_tool_log(tool_logs, log_entry)


def _mcp_company_enrichment_tools() -> tuple[str, ...]:
    configured = os.environ.get("POLICYCHAIN_MCP_COMPANY_ENRICH_TOOLS", "")
    if configured.strip():
        requested = [tool.strip() for tool in configured.split(",") if tool.strip()]
    else:
        requested = list(DEFAULT_COMPANY_ENRICHMENT_TOOLS)
    allowed = set(CNFINANCIAL_COMPANY_TOOLS) - {"search_stock"}
    return tuple(tool for tool in requested if tool in allowed)


def _remaining_candidate_slots(candidates: list[dict[str, Any]], max_candidates: int) -> int:
    return max(max_candidates - len(_dedupe_companies(candidates)), 0)


def _candidate_from_recall(
    item: dict[str, Any],
    impact: dict[str, Any],
    *,
    impact_id: str,
    tool_name: str,
    tool_call_id: str,
    sector: str = "",
    keyword: str = "",
    react_step: str | int | None = None,
    source_type: str = "cnfinancial_recall",
) -> dict[str, Any]:
    candidate = _normalize_company_candidate(item, impact, industry_segment=sector or None)
    candidate["stock_code"] = _normalize_stock_code(str(candidate.get("stock_code") or ""))
    candidate["candidate_source_tool"] = tool_name
    candidate["candidate_query"] = sector or keyword
    candidate["impact_ids"] = [impact_id]
    candidate["provenance"] = [
        {
            "impact_id": impact_id,
            "sector": sector,
            "keyword": keyword,
            "tool": tool_name,
            "tool_call_id": tool_call_id,
            "react_step": react_step,
            "source_type": source_type,
        }
    ]
    candidate["mcp_evidence"] = _tag_evidence(
        normalize_mcp_evidence(
            item,
            query=sector or keyword,
            server_name=CNFINANCIAL_SERVER,
            tool_name=tool_name,
        ),
        impact_id=impact_id,
        tool_call_id=tool_call_id,
        react_step=react_step,
        source_type=source_type,
    )
    return candidate


def _normalize_stock_code(value: str) -> str:
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", value)
    return match.group(1) if match else value.strip()


def _is_valid_a_share_candidate(candidate: dict[str, Any]) -> bool:
    name = str(candidate.get("company_name") or "").strip()
    code = _normalize_stock_code(str(candidate.get("stock_code") or ""))
    return bool(name and re.fullmatch(r"\d{6}", code))


def _enrich_company_candidate(
    candidate: dict[str, Any],
    invoker: MCPToolInvoker | None,
    tool_logs: list[dict[str, Any]] | None = None,
) -> None:
    stock_code = str(candidate.get("stock_code") or "")
    if not stock_code:
        return
    evidence: list[dict[str, Any]] = []
    for tool_name in _mcp_company_enrichment_tools():
        raw = _invoke_or_empty(
            invoker=invoker,
            server_name=CNFINANCIAL_SERVER,
            tool_name=tool_name,
            arguments=_cnfinancial_company_arguments(tool_name, stock_code, candidate),
            tool_logs=tool_logs,
        )
        normalized = normalize_mcp_evidence(
            raw,
            query=stock_code,
            server_name=CNFINANCIAL_SERVER,
            tool_name=tool_name,
        )
        evidence.extend(normalized)
        if tool_name == "get_company_profile":
            _merge_company_profile(candidate, raw)
    candidate["mcp_evidence"] = [*(candidate.get("mcp_evidence") or []), *_dedupe_evidence(evidence)]


def _append_candidate_summary_log(
    tool_logs: list[dict[str, Any]] | None,
    *,
    impact_id: str,
    raw_count: int,
    dedup_count: int,
    kept_count: int,
) -> None:
    log_entry = _new_tool_log(
        server_name="policychain",
        tool_name="company_candidate_pipeline",
        arguments={"impact_id": impact_id},
    )
    log_entry.update(
        {
            "impact_id": impact_id,
            "status": "ok" if kept_count else "empty",
            "count": kept_count,
            "raw_count": raw_count,
            "dedup_count": dedup_count,
            "truncated_count": max(dedup_count - kept_count, 0),
            "error": "",
        }
    )
    _append_tool_log(tool_logs, log_entry)
    _record_mcp_event(log_entry)
    try:
        from policychain.observability import record_event

        payload = {key: value for key, value in log_entry.items() if key not in {"stage", "status"}}
        record_event("candidate.pipeline", stage="company_matcher", status=str(log_entry["status"]), **payload)
    except Exception:
        pass


def _cnfinancial_company_arguments(tool_name: str, stock_code: str, candidate: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "get_company_announcements":
        return {"symbol": stock_code, "num_results": 10}
    if tool_name == "get_competitors":
        return {"symbol": stock_code, "industry": str(candidate.get("industry_segment") or "")}
    return {"symbol": stock_code}


def _normalize_company_candidate(
    item: dict[str, Any],
    impact: dict[str, Any],
    industry_segment: str | None = None,
) -> dict[str, Any]:
    name = _first_value(item, COMPANY_NAME_KEYS)
    code = _first_value(item, COMPANY_CODE_KEYS)
    industry = str(industry_segment or impact.get("chain_segment") or impact.get("industry") or item.get("industry") or "")
    return {
        "company_name": str(name or ""),
        "stock_code": str(code or ""),
        "industry_segment": industry,
        "chain_segment": str(impact.get("chain_segment") or industry),
        "matched_business": str(_first_value(item, BUSINESS_KEYS) or ""),
        "business_keywords": _company_keywords(impact),
        "source_name": "CNFinancial MCP",
        "source_url": str(_first_value(item, ("source_url", "url", "link")) or ""),
        "business_evidence": str(_first_value(item, ("business_evidence", *BUSINESS_KEYS)) or ""),
        "data_date": str(_first_value(item, ("data_date", "date", "publish_date")) or ""),
        "revenue_relevance": "unknown",
        "cnfinancial_raw": item,
    }


def _merge_company_profile(candidate: dict[str, Any], raw: Any) -> None:
    for item in _payload_items(raw):
        business = _first_value(item, BUSINESS_KEYS)
        revenue = _first_value(item, REVENUE_KEYS)
        if business and not candidate.get("matched_business"):
            candidate["matched_business"] = str(business)
        if business and not candidate.get("business_evidence"):
            candidate["business_evidence"] = str(business)
        if revenue:
            candidate["revenue_relevance"] = str(revenue)


def _company_keywords(impact: dict[str, Any]) -> list[str]:
    values = [
        str(impact.get("industry") or ""),
        str(impact.get("chain_segment") or ""),
        str(impact.get("transmission_logic") or ""),
        " ".join(impact.get("business_variables") or []),
        " ".join(impact.get("affected_company_types") or []),
    ]
    text = " ".join(values)
    return [keyword for keyword in COMPANY_BUSINESS_EVIDENCE_KEYWORDS if keyword in text]


def _dedupe_companies(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _dedupe_and_merge_companies(candidates)


def _dedupe_and_merge_companies(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for candidate in candidates:
        key = _company_identity(candidate)
        if key not in by_identity:
            by_identity[key] = candidate
            order.append(key)
            continue
        _merge_company_candidate(by_identity[key], candidate)
    return [by_identity[key] for key in order]


def _company_identity(candidate: dict[str, Any]) -> tuple[str, str]:
    code = _normalize_stock_code(str(candidate.get("stock_code") or ""))
    name = re.sub(r"\s+", "", str(candidate.get("company_name") or "")).lower()
    return ("code", code) if code else ("name", name)


def _merge_company_candidate(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "company_name",
        "stock_code",
        "industry_segment",
        "chain_segment",
        "matched_business",
        "source_url",
        "business_evidence",
        "data_date",
    ):
        if not target.get(key) and source.get(key):
            target[key] = source[key]
    for key in ("impact_ids", "business_keywords", "selected_industries", "selected_concepts"):
        target[key] = _unique([*(target.get(key) or []), *(source.get(key) or [])])
    target["provenance"] = _dedupe_dicts([*(target.get("provenance") or []), *(source.get("provenance") or [])])
    target["mcp_evidence"] = _dedupe_evidence([*(target.get("mcp_evidence") or []), *(source.get("mcp_evidence") or [])])


def _dedupe_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for item in items:
        key = repr(sorted(item.items()))
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def _dedupe_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for item in items:
        key = (
            str(item.get("server_name") or ""),
            str(item.get("tool_name") or ""),
            str(item.get("source_url") or item.get("title") or item.get("query") or ""),
        )
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _clip(text: str, max_chars: int = 500) -> str:
    compacted = re.sub(r"\s+", " ", text).strip()
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 1].rstrip() + "..."
