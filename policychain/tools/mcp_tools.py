from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import os
import re
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from policychain.mcp import (
    MCPToolInvoker,
    MCPToolUnavailable,
    RuntimeMCPInvoker,
    UnavailableMCPInvoker,
    mcp_payload_error_message,
    runtime_mcp_invoker,
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
    "current_name",
    "name",
    "stock_name",
    "sec_name",
    "secName",
    "security_name",
    "\u540d\u79f0",
    "\u80a1\u7968\u7b80\u79f0",
    "\u8bc1\u5238\u7b80\u79f0",
)
COMPANY_CODE_KEYS = (
    "stock_code",
    "code",
    "symbol",
    "sec_code",
    "secCode",
    "security_code",
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

GENERIC_STOCK_SEARCH_TERMS = {
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
    "供应商",
    "服务商",
    "制造商",
    "运营商",
}
COMPANY_SEARCH_STOCK_MAX_CALLS_PER_IMPACT = 2
INVALID_STOCK_SEARCH_MARKERS = (
    "->",
    "=>",
    "→",
    "⇒",
    "⟶",
    "政策措施",
    "政策要求",
    "传导路径",
    "影响路径",
)
INVALID_STOCK_SEARCH_ACTIONS = (
    "推动",
    "促进",
    "支持",
    "要求",
    "实施",
    "建设",
    "形成",
    "影响",
    "带动",
    "增加",
    "降低",
    "提升",
    "采购",
    "布局",
    "开展",
    "组织",
)
INVALID_STOCK_SEARCH_METRIC_TERMS = (
    "收入确认",
    "收入",
    "营收",
    "销量",
    "单价",
    "价格",
    "投资额",
    "金额",
    "资本开支",
    "成本",
    "利润",
    "毛利",
    "产量",
    "装机量",
    "渗透率",
    "利用率",
    "交易量",
    "运营效率",
    "效率",
    "能效",
    "碳效",
    "配套率",
    "增长率",
    "市占率",
    "份额",
    "确认节奏",
    "节奏",
    "应用场景",
    "场景",
    "指标",
    "经营变量",
)
GENERIC_TERM_SUFFIXES = (
    "需求",
    "规模",
    "成本",
    "价格",
    "收入",
    "销量",
    "产量",
    "装机量",
    "渗透率",
    "利用率",
    "资本开支",
    "订单",
    "能耗",
    "企业",
    "公司",
    "供应商",
    "服务商",
    "制造商",
    "运营商",
    "机构",
    "行业",
    "产业",
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
    active_logs = tool_logs if tool_logs is not None else []
    industry_catalog, concept_catalog = _load_cnfinancial_sector_catalogs(invoker, active_logs)
    catalog_logs = list(active_logs[-2:])
    for log in catalog_logs:
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
        _append_sector_selection_log(active_logs, selection, impact_id=impact_id)
        legal_industries = _selected_sector_names(selection, "selected_industries")
        selected_concepts = _selected_sector_names(selection, "selected_concepts")
        recalled: list[dict[str, Any]] = []
        recall_logs: list[dict[str, Any]] = []

        for industry in legal_industries:
            raw, call_log = _invoke_with_log(
                invoker=invoker,
                server_name=CNFINANCIAL_SERVER,
                tool_name="get_industry_stocks",
                arguments={"industry": industry},
                tool_logs=active_logs,
                log_context={"impact_id": impact_id, "sector": industry, "source_type": "raw_recall"},
            )
            recall_logs.append(call_log)
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

        search_term_budget = _mcp_int_setting("POLICYCHAIN_MCP_MAX_SEARCH_TERMS", 2, minimum=0)
        search_terms = _limited(
            [
                term
                for term in _unique(
                    [
                        *_candidate_stock_search_terms(impact),
                        *selected_concepts,
                        *legal_industries,
                    ]
                )
                if _is_specific_stock_search_term(term)
            ],
            search_term_budget,
        )
        for term in search_terms:
            raw, call_log = _invoke_with_log(
                invoker=invoker,
                server_name=CNFINANCIAL_SERVER,
                tool_name="search_stock",
                arguments={"keyword": term},
                tool_logs=active_logs,
                log_context={"impact_id": impact_id, "keyword": term, "source_type": "raw_recall"},
            )
            recall_logs.append(call_log)
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
        for candidate in kept:
            candidate["retrieval_status"] = "ok"
        _append_candidate_summary_log(
            active_logs,
            impact_id=impact_id,
            raw_count=len(recalled),
            dedup_count=len(deduped),
            kept_count=len(kept),
            recall_logs=recall_logs or catalog_logs,
            query_terms=search_terms,
        )
        path_candidates.extend(kept)

    candidates = _dedupe_and_merge_companies(path_candidates)
    for candidate in candidates:
        _enrich_company_candidate(candidate, invoker=invoker, tool_logs=active_logs)
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


def resolve_company_seeds(
    seeds: list[dict[str, Any]],
    industry_impacts: list[dict[str, Any]],
    invoker: MCPToolInvoker | None = None,
    tool_logs: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve untrusted LLM/Web seeds into identity-verified CNFinancial candidates.

    Search results are clues only. A seed is promoted only when a current A-share
    identity is confirmed by an explicit current-status company-info response or
    by an official exchange/CNInfo/company disclosure. Company profile data may
    enrich a known code, but never proves that the security is currently listed.
    """

    impacts = {
        _impact_identifier(impact, index): impact
        for index, impact in enumerate(industry_impacts, start=1)
    }
    active_logs = tool_logs if tool_logs is not None else []
    audit: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []
    search_cache: dict[str, tuple[Any, dict[str, Any]]] = {}

    for seed_index, seed in enumerate(seeds, start=1):
        impact_id = str(seed.get("impact_id") or "")
        seed_id = str(seed.get("seed_id") or f"seed-{seed_index:03d}")
        proposed_name = str(seed.get("proposed_name") or "").strip()
        historical_names = _unique(str(value).strip() for value in (seed.get("historical_names") or []))[:3]
        proposed_code = _normalize_stock_code(str(seed.get("proposed_stock_code") or ""))
        base = {
            "time": datetime.now(timezone.utc).isoformat(),
            "seed_id": seed_id,
            "impact_id": impact_id,
            "proposed_name": proposed_name,
            "proposed_stock_code": proposed_code,
            "source": ",".join(str(value) for value in (seed.get("origin_channels") or [])) or "llm",
            "url": "",
            "date": str(seed.get("time") or ""),
            "tool_call_id": str(seed.get("tool_call_id") or ""),
            "cache_hit": False,
        }
        if impact_id not in impacts:
            _append_company_seed_audit(audit, base, "rejected", "invalid_impact_id")
            continue
        if not proposed_name:
            _append_company_seed_audit(audit, base, "rejected", "missing_proposed_name")
            continue
        if proposed_code and not _is_current_a_share_code(proposed_code):
            _append_company_seed_audit(audit, base, "rejected", "non_current_a_share_code")
            continue

        search_items: list[dict[str, Any]] = []
        search_logs: list[dict[str, Any]] = []
        for exact_name in _unique([proposed_name, *historical_names]):
            cache_key = _normalize_company_name(exact_name)
            cache_hit = cache_key in search_cache
            if cache_hit:
                raw, call_log = search_cache[cache_key]
            else:
                raw, call_log = _invoke_with_log(
                    invoker=invoker,
                    server_name=CNFINANCIAL_SERVER,
                    tool_name="search_stock",
                    arguments={"keyword": exact_name},
                    tool_logs=active_logs,
                    log_context={
                        "impact_id": impact_id,
                        "seed_id": seed_id,
                        "source_type": "company_seed_identity_clue",
                    },
                )
                search_cache[cache_key] = (raw, call_log)
            search_items.extend(_payload_items(raw))
            search_logs.append({**call_log, "cache_hit": cache_hit or bool(call_log.get("cache_hit"))})

        search_codes = _unique(
            _normalize_stock_code(str(_first_value(item, COMPANY_CODE_KEYS) or ""))
            for item in search_items
            if _normalize_stock_code(str(_first_value(item, COMPANY_CODE_KEYS) or ""))
        )
        search_names_by_code: dict[str, list[str]] = {}
        for item in search_items:
            item_code = _normalize_stock_code(str(_first_value(item, COMPANY_CODE_KEYS) or ""))
            item_name = str(_first_value(item, COMPANY_NAME_KEYS) or "").strip()
            if item_code and item_name:
                search_names_by_code.setdefault(item_code, [])
                search_names_by_code[item_code] = _unique([*search_names_by_code[item_code], item_name])

        base["tool_call_id"] = str(search_logs[0].get("tool_call_id") or base["tool_call_id"]) if search_logs else base["tool_call_id"]
        base["cache_hit"] = bool(search_logs and all(bool(item.get("cache_hit")) for item in search_logs))
        if len(search_codes) > 1:
            _append_company_seed_audit(audit, base, "rejected", "ambiguous_name_multiple_codes")
            continue
        search_code = search_codes[0] if search_codes else ""
        if proposed_code and search_code and proposed_code != search_code:
            _append_company_seed_audit(audit, base, "rejected", "name_code_conflict")
            continue
        resolved_code = proposed_code or search_code
        if not resolved_code:
            reason_code = _search_failure_reason(search_logs, fallback="identity_code_unresolved")
            _append_company_seed_audit(audit, base, "unresolved", reason_code)
            continue
        if not _is_current_a_share_code(resolved_code):
            _append_company_seed_audit(audit, base, "rejected", "non_current_a_share_code")
            continue
        prepared.append(
            {
                "seed": seed,
                "base": base,
                "code": resolved_code,
                "search_items": search_items,
                "search_names": search_names_by_code.get(resolved_code, []),
                "search_empty": not search_items,
                "search_logs": search_logs,
            }
        )

    codes_by_name: dict[str, set[str]] = {}
    for item in prepared:
        name_key = _normalize_company_name(str(item["seed"].get("proposed_name") or ""))
        codes_by_name.setdefault(name_key, set()).add(str(item["code"]))
    ambiguous_seed_names = {name for name, codes in codes_by_name.items() if name and len(codes) > 1}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in prepared:
        name_key = _normalize_company_name(str(item["seed"].get("proposed_name") or ""))
        if name_key in ambiguous_seed_names:
            _append_company_seed_audit(
                audit,
                item["base"],
                "rejected",
                "ambiguous_name_multiple_codes",
            )
            continue
        grouped.setdefault(str(item["code"]), []).append(item)

    candidates: list[dict[str, Any]] = []
    for code, group in grouped.items():
        first = group[0]
        first_seed = first["seed"]
        first_base = first["base"]
        profile_raw, profile_log = _invoke_with_log(
            invoker=invoker,
            server_name=CNFINANCIAL_SERVER,
            tool_name="get_company_profile",
            arguments={"symbol": code},
            tool_logs=active_logs,
            log_context={
                "impact_id": str(first_seed.get("impact_id") or ""),
                "seed_id": str(first_seed.get("seed_id") or ""),
                "source_type": "company_seed_enrichment",
            },
        )
        info_raw, info_log = _invoke_with_log(
            invoker=invoker,
            server_name=CNFINANCIAL_SERVER,
            tool_name="get_company_info",
            arguments={"symbol": code},
            tool_logs=active_logs,
            log_context={
                "impact_id": str(first_seed.get("impact_id") or ""),
                "seed_id": str(first_seed.get("seed_id") or ""),
                "source_type": "company_seed_identity",
            },
        )
        identity_query = (
            f'{str(first_seed.get("proposed_name") or "")} {code} '
            "证券简称 证券代码 当前上市 公司概况 更名 公告"
        ).strip()
        web_raw, web_log = _invoke_with_log(
            invoker=invoker,
            server_name=OPEN_WEBSEARCH_SERVER,
            tool_name=OPEN_WEBSEARCH_SEARCH_TOOL,
            arguments={"query": identity_query, "limit": 5},
            tool_logs=active_logs,
            log_context={
                "impact_id": str(first_seed.get("impact_id") or ""),
                "seed_id": str(first_seed.get("seed_id") or ""),
                "source_type": "company_seed_official_identity",
            },
        )
        web_evidence = normalize_mcp_evidence(
            web_raw,
            query=identity_query,
            server_name=OPEN_WEBSEARCH_SERVER,
            tool_name=OPEN_WEBSEARCH_SEARCH_TOOL,
            source_priority=COMPANY_SOURCE_PRIORITY,
        )
        candidate_names = _unique(
            [
                *(name for item in group for name in item.get("search_names") or []),
                *(str(item["seed"].get("proposed_name") or "") for item in group),
                *(str(name) for item in group for name in (item["seed"].get("historical_names") or [])),
            ]
        )
        identity = _resolve_current_company_identity(
            code=code,
            candidate_names=candidate_names,
            company_info=info_raw,
            web_evidence=web_evidence,
        )
        if str(identity.get("status") or "") == "unresolved":
            identity["reason_code"] = _search_failure_reason(
                [info_log, web_log],
                fallback=str(identity.get("reason_code") or "current_identity_unverified"),
            )

        accepted: list[dict[str, Any]] = []
        official_url = str(identity.get("source_url") or "")
        official_date = str(identity.get("data_date") or "")
        for group_index, item in enumerate(group):
            seed = item["seed"]
            base = {
                **item["base"],
                "stock_code": code,
                "company_name": str(identity.get("company_name") or ""),
                "tool_call_id": str(identity.get("tool_call_id") or info_log.get("tool_call_id") or web_log.get("tool_call_id") or ""),
                "source": str(identity.get("source") or "official_identity"),
                "url": official_url,
                "date": official_date,
                "cache_hit": group_index > 0 or bool(identity.get("cache_hit")),
            }
            enrichment_reason = (
                "profile_enriched"
                if _payload_items(profile_raw)
                else _search_failure_reason([profile_log], fallback="profile_empty")
            )
            _record_company_stage_event(
                "company.enrichment",
                base={
                    **base,
                    "tool_call_id": str(profile_log.get("tool_call_id") or ""),
                    "source": CNFINANCIAL_SERVER,
                    "cache_hit": group_index > 0 or bool(profile_log.get("cache_hit")),
                },
                status="ok" if _payload_items(profile_raw) else str(profile_log.get("status") or "empty"),
                reason_code=enrichment_reason,
            )
            if str(identity.get("status") or "") != "verified":
                _append_company_seed_audit(
                    audit,
                    base,
                    str(identity.get("status") or "unresolved"),
                    str(identity.get("reason_code") or "current_identity_unverified"),
                )
                continue
            proposed_name = str(seed.get("proposed_name") or "")
            current_name = str(identity.get("company_name") or "")
            if _normalize_company_name(proposed_name) != _normalize_company_name(current_name):
                if not _official_rename_chain(
                    code=code,
                    old_name=proposed_name,
                    current_name=current_name,
                    evidence=web_evidence,
                ):
                    _append_company_seed_audit(audit, base, "rejected", "unverified_or_ambiguous_alias")
                    continue
            reason_code = "identity_verified"
            if bool(item.get("search_empty")) and _payload_items(profile_raw):
                reason_code = "profile_found_search_empty"
            _append_company_seed_audit(audit, base, "verified", reason_code)
            accepted.append(item)

        if not accepted:
            continue

        impact_ids = _unique(str(item["seed"].get("impact_id") or "") for item in accepted)
        first_impact = impacts[impact_ids[0]]
        profile_evidence = normalize_mcp_evidence(
            profile_raw,
            query=code,
            server_name=CNFINANCIAL_SERVER,
            tool_name="get_company_profile",
        )
        info_evidence = normalize_mcp_evidence(
            info_raw,
            query=code,
            server_name=CNFINANCIAL_SERVER,
            tool_name="get_company_info",
        )
        business_values = _unique(
            str(_first_value(raw_item, BUSINESS_KEYS) or "")
            for raw_item in [*_payload_items(profile_raw), *_payload_items(info_raw)]
        )
        data_dates = _unique(
            str(_first_value(raw_item, ("data_date", "date", "publish_date", "year")) or "")
            for raw_item in [*_payload_items(profile_raw), *_payload_items(info_raw)]
        )
        candidate = {
            "company_name": str(identity.get("company_name") or ""),
            "stock_code": code,
            "industry_segment": str(first_impact.get("chain_segment") or first_impact.get("industry") or ""),
            "chain_segment": str(first_impact.get("chain_segment") or first_impact.get("industry") or ""),
            "matched_business": " ".join(business_values),
            "business_evidence": " ".join(business_values),
            "business_keywords": _unique(
                keyword
                for impact_id in impact_ids
                for keyword in _company_keywords(impacts[impact_id])
            ),
            "source_name": "CNFinancial MCP + official identity evidence",
            "source_url": official_url,
            "data_date": data_dates[0] if data_dates else official_date or "unknown",
            "revenue_relevance": "unknown",
            "candidate_source_tool": "get_company_profile",
            "impact_ids": impact_ids,
            "seed_reasons": _unique(str(item["seed"].get("seed_reason") or "") for item in accepted),
            "identity_verified": True,
            "provenance": [
                {
                    "impact_id": str(item["seed"].get("impact_id") or ""),
                    "seed_id": str(item["seed"].get("seed_id") or ""),
                    "seed_reason": str(item["seed"].get("seed_reason") or ""),
                    "origin_channels": list(item["seed"].get("origin_channels") or []),
                    "tool": "get_company_profile",
                    "tool_call_id": str(profile_log.get("tool_call_id") or ""),
                    "source_type": "llm_web_seed_cnfinancial_verified",
                    "source_url": official_url,
                    "data_date": official_date,
                }
                for item in accepted
            ],
            "mcp_evidence": _dedupe_evidence([*profile_evidence, *info_evidence, *web_evidence]),
            "cnfinancial_raw": _payload_items(profile_raw)[0] if _payload_items(profile_raw) else {},
        }
        candidates.append(candidate)

    return _dedupe_and_merge_companies(candidates), audit


def _append_company_seed_audit(
    audit: list[dict[str, Any]],
    base: dict[str, Any],
    status: str,
    reason_code: str,
) -> None:
    entry = {**base, "status": status, "reason_code": reason_code}
    audit.append(entry)
    _record_company_stage_event("company.identity", base=base, status=status, reason_code=reason_code)


def _record_company_stage_event(
    event_type: str,
    *,
    base: dict[str, Any],
    status: str,
    reason_code: str,
) -> None:
    try:
        from policychain.observability import record_event

        record_event(
            event_type,
            stage="company_matcher",
            status=status,
            seed_id=str(base.get("seed_id") or ""),
            impact_id=str(base.get("impact_id") or ""),
            tool_call_id=str(base.get("tool_call_id") or ""),
            reason_code=reason_code,
            cache_hit=bool(base.get("cache_hit")),
            source=str(base.get("source") or ""),
            url=str(base.get("url") or ""),
            date=str(base.get("date") or ""),
            company_name=str(base.get("company_name") or ""),
            stock_code=str(base.get("stock_code") or base.get("proposed_stock_code") or ""),
        )
    except Exception:
        return


def _resolve_current_company_identity(
    *,
    code: str,
    candidate_names: list[str],
    company_info: Any,
    web_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    identities: list[dict[str, Any]] = []
    non_current = False
    for item in _payload_items(company_info):
        item_code = _normalize_stock_code(str(_first_value(item, COMPANY_CODE_KEYS) or code))
        if item_code != code:
            continue
        status_value = _first_value(
            item,
            (
                "listing_status",
                "list_status",
                "listed_status",
                "trade_status",
                "market_status",
                "status",
                "is_listed",
                "listed",
                "上市状态",
                "交易状态",
            ),
        )
        if _is_non_current_listing_status(status_value):
            non_current = True
            continue
        name = str(_first_value(item, COMPANY_NAME_KEYS) or "").strip()
        if name and _is_current_listing_status(status_value):
            identities.append(
                {
                    "company_name": name,
                    "source": CNFINANCIAL_SERVER,
                    "source_url": str(_first_value(item, ("source_url", "url", "link")) or ""),
                    "data_date": str(_first_value(item, ("data_date", "date", "publish_date")) or ""),
                    "tool_call_id": "",
                }
            )

    for evidence in web_evidence:
        raw_item = evidence.get("raw_payload") if isinstance(evidence.get("raw_payload"), dict) else {}
        source_url = str(evidence.get("source_url") or "")
        source_org = str(evidence.get("source_org") or "")
        if not _is_official_company_identity_source(source_url, source_org):
            continue
        text = " ".join(
            str(value or "")
            for value in (
                evidence.get("title"),
                evidence.get("summary"),
                raw_item.get("content"),
                raw_item.get("description"),
                raw_item.get("text"),
                raw_item.get("stock_code"),
                raw_item.get("secCode"),
                raw_item.get("status"),
                raw_item.get("listing_status"),
                raw_item.get("current_name"),
                raw_item.get("old_name"),
                raw_item.get("former_name"),
            )
        )
        raw_code = _normalize_stock_code(str(_first_value(raw_item, COMPANY_CODE_KEYS) or ""))
        if raw_code and raw_code != code:
            continue
        if not raw_code and code not in text:
            continue
        raw_status = _first_value(
            raw_item,
            (
                "listing_status",
                "list_status",
                "listed_status",
                "trade_status",
                "market_status",
                "status",
                "is_listed",
                "listed",
                "上市状态",
                "交易状态",
            ),
        )
        if _contains_non_current_listing_marker(text) or _is_non_current_listing_status(raw_status):
            non_current = True
            continue
        if not _contains_current_identity_marker(text) and not _is_current_listing_status(raw_status):
            continue
        explicit_name = str(_first_value(raw_item, COMPANY_NAME_KEYS) or "").strip()
        matched_names = [name for name in candidate_names if _normalize_company_name(name) in _normalize_company_name(text)]
        name = explicit_name or _declared_current_name(text, matched_names)
        if not name and len({_normalize_company_name(value) for value in matched_names}) == 1:
            name = matched_names[0]
        if name:
            identities.append(
                {
                    "company_name": name,
                    "source": source_org or "official_web",
                    "source_url": source_url,
                    "data_date": str(evidence.get("published_date") or ""),
                    "tool_call_id": "",
                }
            )

    if non_current:
        return {"status": "rejected", "reason_code": "non_current_a_share_identity"}
    by_name: dict[str, dict[str, Any]] = {}
    for identity in identities:
        normalized_name = _normalize_company_name(str(identity.get("company_name") or ""))
        if normalized_name:
            by_name.setdefault(normalized_name, identity)
    if not by_name:
        return {"status": "unresolved", "reason_code": "current_identity_unverified"}
    if len(by_name) > 1:
        return {"status": "rejected", "reason_code": "ambiguous_current_identity"}
    return {"status": "verified", "reason_code": "identity_verified", **next(iter(by_name.values()))}


def _official_rename_chain(
    *,
    code: str,
    old_name: str,
    current_name: str,
    evidence: list[dict[str, Any]],
) -> bool:
    old_key = _normalize_company_name(old_name)
    current_key = _normalize_company_name(current_name)
    if not old_key or not current_key or old_key == current_key:
        return old_key == current_key
    for item in evidence:
        if not _is_official_company_identity_source(str(item.get("source_url") or ""), str(item.get("source_org") or "")):
            continue
        raw_item = item.get("raw_payload") if isinstance(item.get("raw_payload"), dict) else {}
        text = _normalize_company_name(
            " ".join(
                str(value or "")
                for value in (
                    item.get("title"),
                    item.get("summary"),
                    raw_item.get("content"),
                    raw_item.get("description"),
                    raw_item.get("text"),
                    raw_item.get("stock_code"),
                    raw_item.get("secCode"),
                    raw_item.get("current_name"),
                    raw_item.get("old_name"),
                    raw_item.get("former_name"),
                )
            )
        )
        if code in text and old_key in text and current_key in text and re.search(r"更名|曾用名|原名|证券简称变更|股票简称变更", text):
            return True
    return False


def _declared_current_name(text: str, candidate_names: list[str]) -> str:
    normalized_text = _normalize_company_name(text)
    declared = _unique(
        name
        for name in candidate_names
        if any(
            f"{marker}{_normalize_company_name(name)}" in normalized_text
            for marker in ("变更为", "更名为", "现名为", "当前证券简称", "证券简称为", "股票简称为")
        )
    )
    return declared[0] if len(declared) == 1 else ""


def _is_official_company_identity_source(source_url: str, source_org: str) -> bool:
    source_text = f"{source_url} {source_org}".lower()
    return any(
        marker in source_text
        for marker in (
            "cninfo.com.cn",
            "sse.com.cn",
            "szse.cn",
            "bse.cn",
            "巨潮资讯",
            "上海证券交易所",
            "深圳证券交易所",
            "北京证券交易所",
            "公司官网",
            "official",
        )
    )


def _contains_current_identity_marker(text: str) -> bool:
    return bool(re.search(r"证券简称|股票简称|证券代码|股票代码|公司概况|当前上市|正常上市|上市公司|当前交易", text))


def _contains_non_current_listing_marker(text: str) -> bool:
    return bool(re.search(r"终止上市|退市|摘牌|暂停上市|已退市|delisted|terminated", text, re.IGNORECASE))


def _is_current_listing_status(value: Any) -> bool:
    if value is True:
        return True
    text = str(value or "").strip().lower()
    if not text or _is_non_current_listing_status(text):
        return False
    return text in {"active", "listed", "normal", "trading", "上市", "正常", "交易", "正常上市"} or any(
        marker in text for marker in ("正常交易", "当前上市", "正常上市")
    )


def _is_non_current_listing_status(value: Any) -> bool:
    if value is False:
        return True
    text = str(value or "").strip().lower()
    return any(marker in text for marker in ("delisted", "terminated", "inactive", "退市", "摘牌", "终止上市", "暂停上市"))


def _is_current_a_share_code(code: str) -> bool:
    if not re.fullmatch(r"\d{6}", code) or code.startswith(("200", "900")):
        return False
    return code.startswith(("0", "3", "4", "6", "8", "920"))


def _normalize_company_name(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]", "", str(value or "")).lower()


def _search_failure_reason(logs: list[dict[str, Any]], fallback: str) -> str:
    statuses = {str(item.get("status") or "") for item in logs}
    if "unavailable" in statuses:
        return "tool_unavailable"
    if "error" in statuses:
        return "tool_error"
    if "skipped" in statuses:
        return "tool_skipped"
    return fallback


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


def collect_company_discovery_web_evidence(
    impact_id: str,
    queries: list[str],
    invoker: MCPToolInvoker | None = None,
    top_k: int = 5,
    tool_logs: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the bounded Web-first discovery channel for one impact path."""

    evidence: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for query in _unique(str(value).strip() for value in queries if str(value).strip())[:2]:
        raw, call_log = _invoke_with_log(
            invoker=invoker,
            server_name=OPEN_WEBSEARCH_SERVER,
            tool_name=OPEN_WEBSEARCH_SEARCH_TOOL,
            arguments={"query": query, "limit": max(min(int(top_k), 8), 1)},
            tool_logs=tool_logs,
            log_context={
                "stage": "company_matcher",
                "impact_id": impact_id,
                "source_type": "company_web_discovery",
            },
        )
        normalized = _tag_evidence(
            normalize_mcp_evidence(
                raw,
                query=query,
                server_name=OPEN_WEBSEARCH_SERVER,
                tool_name=OPEN_WEBSEARCH_SEARCH_TOOL,
                source_priority=COMPANY_SOURCE_PRIORITY,
            ),
            impact_id=impact_id,
            source_type="company_web_discovery",
            tool_call_id=str(call_log.get("tool_call_id") or ""),
        )
        evidence.extend(normalized)
        status = str(call_log.get("status") or "empty")
        reason_code = "web_results" if normalized else _web_discovery_reason(status)
        audit.append(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "impact_id": impact_id,
                "seed_id": "",
                "tool_call_id": str(call_log.get("tool_call_id") or ""),
                "source": OPEN_WEBSEARCH_SERVER,
                "reason_code": reason_code,
                "cache_hit": bool(call_log.get("cache_hit")),
                "status": "ok" if normalized else status,
                "query": query,
                "count": len(normalized),
            }
        )
    return _dedupe_evidence(evidence), audit


def resolve_web_first_company_seeds(
    seeds: list[dict[str, Any]],
    industry_impacts: list[dict[str, Any]],
    discovery_evidence: list[dict[str, Any]],
    invoker: MCPToolInvoker | None = None,
    tool_logs: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve explicit Web/LLM seeds without industry or concept candidate recall.

    CNFinancial is queried only with an explicit company name or six-digit code.
    A technical CNFinancial failure may use a strict two-source Web fallback;
    a successful empty response never activates that fallback.
    """

    impacts = {
        _impact_identifier(impact, index): {**impact, "impact_id": _impact_identifier(impact, index)}
        for index, impact in enumerate(industry_impacts, start=1)
    }
    evidence_by_impact: dict[str, list[dict[str, Any]]] = {}
    for item in discovery_evidence:
        evidence_by_impact.setdefault(str(item.get("impact_id") or ""), []).append(item)
    active_logs = tool_logs if tool_logs is not None else []
    audit: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    search_cache: dict[str, tuple[Any, dict[str, Any]]] = {}
    prepared: list[dict[str, Any]] = []

    for seed_index, seed in enumerate(seeds, start=1):
        impact_id = str(seed.get("impact_id") or "")
        seed_id = str(seed.get("seed_id") or f"seed-{seed_index:03d}")
        proposed_name = str(seed.get("proposed_name") or "").strip()
        proposed_code = _normalize_stock_code(str(seed.get("proposed_stock_code") or ""))
        base = _web_first_audit_base(seed, seed_id=seed_id, proposed_code=proposed_code)
        if impact_id not in impacts:
            _append_company_seed_audit(audit, base, "rejected", "invalid_impact_id")
            continue
        if not proposed_name:
            _append_company_seed_audit(audit, base, "rejected", "missing_proposed_name")
            continue
        if proposed_code and not _is_current_a_share_code(proposed_code):
            _append_company_seed_audit(audit, base, "rejected", "non_current_a_share_code")
            continue

        search_items: list[dict[str, Any]] = []
        search_logs: list[dict[str, Any]] = []
        search_names = [proposed_name]
        aliases = _unique(str(value).strip() for value in (seed.get("historical_names") or []) if str(value).strip())
        if aliases:
            search_names.append(aliases[0])
        for name_index, exact_name in enumerate(search_names):
            if name_index and search_items:
                break
            cache_key = _normalize_company_name(exact_name)
            cache_hit = cache_key in search_cache
            if cache_hit:
                raw, call_log = search_cache[cache_key]
            else:
                raw, call_log = _invoke_with_log(
                    invoker=invoker,
                    server_name=CNFINANCIAL_SERVER,
                    tool_name="search_stock",
                    arguments={"keyword": exact_name},
                    tool_logs=active_logs,
                    log_context={
                        "stage": "company_matcher",
                        "impact_id": impact_id,
                        "seed_id": seed_id,
                        "source_type": "company_exact_identity",
                    },
                )
                search_cache[cache_key] = (raw, call_log)
            search_logs.append({**call_log, "cache_hit": cache_hit or bool(call_log.get("cache_hit"))})
            search_items.extend(_payload_items(raw))

        explicit_mappings: list[tuple[str, str]] = []
        for item in search_items:
            item_code = _normalize_stock_code(str(_first_value(item, COMPANY_CODE_KEYS) or ""))
            item_name = str(_first_value(item, COMPANY_NAME_KEYS) or "").strip()
            if item_code and item_name:
                explicit_mappings.append((item_name, item_code))
        search_codes = _unique(code for _name, code in explicit_mappings)
        base["tool_call_id"] = str(search_logs[0].get("tool_call_id") or base["tool_call_id"]) if search_logs else base["tool_call_id"]
        base["cache_hit"] = bool(search_logs and all(bool(item.get("cache_hit")) for item in search_logs))
        if len(search_codes) > 1:
            _append_company_seed_audit(audit, base, "rejected", "identity_conflict")
            continue
        search_code = search_codes[0] if search_codes else ""
        if proposed_code and search_code and proposed_code != search_code:
            _append_company_seed_audit(audit, base, "rejected", "identity_conflict")
            continue

        search_reason = _cnfinancial_log_reason(search_logs)
        web_code, web_code_conflict = _web_identity_code(
            evidence_by_impact.get(impact_id, []),
            proposed_name,
        )
        if web_code_conflict:
            _append_company_seed_audit(audit, base, "rejected", "identity_conflict")
            continue
        resolved_code = proposed_code or search_code
        if not resolved_code and search_reason in {"cnfinancial_error", "cnfinancial_unavailable"}:
            resolved_code = web_code
        if not resolved_code:
            _append_company_seed_audit(audit, base, "unresolved", search_reason or "cnfinancial_empty")
            continue
        if not _is_current_a_share_code(resolved_code):
            _append_company_seed_audit(audit, base, "rejected", "non_current_a_share_code")
            continue
        prepared.append(
            {
                "seed": seed,
                "base": base,
                "code": resolved_code,
                "search_reason": search_reason,
                "search_items": search_items,
                "search_names": _unique(name for name, code in explicit_mappings if code == resolved_code),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in prepared:
        grouped.setdefault(str(item["code"]), []).append(item)

    candidates: list[dict[str, Any]] = []
    for code, group in grouped.items():
        first_seed = group[0]["seed"]
        first_impact_id = str(first_seed.get("impact_id") or "")
        info_raw, info_log = _invoke_with_log(
            invoker=invoker,
            server_name=CNFINANCIAL_SERVER,
            tool_name="get_company_info",
            arguments={"symbol": code},
            tool_logs=active_logs,
            log_context={
                "stage": "company_matcher",
                "impact_id": first_impact_id,
                "seed_id": str(first_seed.get("seed_id") or ""),
                "source_type": "company_exact_identity",
            },
        )
        profile_raw, profile_log = _invoke_with_log(
            invoker=invoker,
            server_name=CNFINANCIAL_SERVER,
            tool_name="get_company_profile",
            arguments={"symbol": code},
            tool_logs=active_logs,
            log_context={
                "stage": "company_matcher",
                "impact_id": first_impact_id,
                "seed_id": str(first_seed.get("seed_id") or ""),
                "source_type": "company_exact_enrichment",
            },
        )
        candidate_names = _unique(
            [
                *(name for item in group for name in item.get("search_names") or []),
                *(str(item["seed"].get("proposed_name") or "") for item in group),
                *(str(name) for item in group for name in (item["seed"].get("historical_names") or [])[:1]),
            ]
        )
        identity = _resolve_current_company_identity(
            code=code,
            candidate_names=candidate_names,
            company_info=info_raw,
            web_evidence=[],
        )
        info_reason = _cnfinancial_log_reason([info_log])
        profile_reason = _cnfinancial_log_reason([profile_log])
        business_values = _unique(
            str(_first_value(item, BUSINESS_KEYS) or "")
            for item in [*_payload_items(profile_raw), *_payload_items(info_raw)]
        )
        accepted: list[dict[str, Any]] = []
        for group_index, item in enumerate(group):
            seed = item["seed"]
            impact_id = str(seed.get("impact_id") or "")
            impact = impacts[impact_id]
            scoped_web = _company_scoped_web_evidence(
                evidence_by_impact.get(impact_id, []),
                names=[
                    str(seed.get("proposed_name") or ""),
                    str(identity.get("company_name") or ""),
                    *(str(value) for value in (seed.get("historical_names") or [])[:1]),
                ],
                code=code,
            )
            base = {
                **item["base"],
                "company_name": str(identity.get("company_name") or seed.get("proposed_name") or ""),
                "stock_code": code,
                "tool_call_id": str(info_log.get("tool_call_id") or ""),
                "source": CNFINANCIAL_SERVER,
                "url": str(identity.get("source_url") or ""),
                "date": str(identity.get("data_date") or seed.get("time") or ""),
                "cache_hit": group_index > 0,
            }
            _record_company_stage_event(
                "company.enrichment",
                base={
                    **base,
                    "tool_call_id": str(profile_log.get("tool_call_id") or ""),
                    "cache_hit": group_index > 0,
                },
                status=str(profile_log.get("status") or "empty"),
                reason_code="cnfinancial_business" if business_values else profile_reason,
            )

            if _contains_non_current_listing_marker(_web_evidence_text(scoped_web)):
                _append_company_seed_audit(audit, base, "rejected", "non_current_a_share_identity")
                continue
            if str(identity.get("status") or "") == "rejected":
                _append_company_seed_audit(
                    audit,
                    base,
                    "rejected",
                    str(identity.get("reason_code") or "identity_conflict"),
                )
                continue

            normal_verified = str(identity.get("status") or "") == "verified" and bool(business_values)
            current_name = str(identity.get("company_name") or "")
            proposed_name = str(seed.get("proposed_name") or "")
            if normal_verified and _normalize_company_name(proposed_name) != _normalize_company_name(current_name):
                if not _official_rename_chain(
                    code=code,
                    old_name=proposed_name,
                    current_name=current_name,
                    evidence=scoped_web,
                ):
                    _append_company_seed_audit(audit, base, "rejected", "identity_conflict")
                    continue

            web_fallback = False
            fallback_bundle: dict[str, Any] = {}
            if not normal_verified:
                successful_empty = (
                    (str(info_log.get("status") or "") == "empty" and str(identity.get("status") or "") != "verified")
                    or (str(profile_log.get("status") or "") == "empty" and not business_values)
                    or item.get("search_reason") == "cnfinancial_empty"
                )
                technical_failure = any(
                    reason in {"cnfinancial_error", "cnfinancial_unavailable"}
                    for reason in (item.get("search_reason"), info_reason, profile_reason)
                )
                if successful_empty or not technical_failure:
                    reason_code = "cnfinancial_empty" if successful_empty else "business_rejected"
                    _append_company_seed_audit(audit, base, "unresolved", reason_code)
                    bundles.append(
                        _company_evidence_bundle(
                            seed,
                            code,
                            scoped_web,
                            profile_raw,
                            info_raw,
                            status=reason_code,
                            identity=identity,
                            impact=impact,
                            path_specific_business=" ".join(business_values),
                            profile_log=profile_log,
                            info_log=info_log,
                        )
                    )
                    continue
                fallback_bundle = _qualify_web_fallback(seed, code, impact, scoped_web)
                if str(fallback_bundle.get("status") or "") != "verified":
                    _append_company_seed_audit(
                        audit,
                        base,
                        str(fallback_bundle.get("status") or "unresolved"),
                        str(fallback_bundle.get("reason_code") or "web_fallback_unresolved"),
                    )
                    bundles.append(
                        _company_evidence_bundle(
                            seed,
                            code,
                            scoped_web,
                            profile_raw,
                            info_raw,
                            status=str(fallback_bundle.get("reason_code") or "web_fallback_unresolved"),
                            identity=identity,
                            impact=impact,
                            path_specific_business=str(fallback_bundle.get("business_text") or ""),
                            profile_log=profile_log,
                            info_log=info_log,
                        )
                    )
                    continue
                web_fallback = True

            reason_code = "web_fallback" if web_fallback else "identity_verified"
            _append_company_seed_audit(audit, base, "verified", reason_code)
            accepted.append(
                {
                    "item": item,
                    "company_name": proposed_name if web_fallback else current_name,
                    "business_text": str(fallback_bundle.get("business_text") or "") if web_fallback else " ".join(business_values),
                    "web_fallback": web_fallback,
                    "web_evidence": list(fallback_bundle.get("evidence") or scoped_web),
                    "path_terms": list(fallback_bundle.get("path_terms") or []),
                }
            )
            bundles.append(
                _company_evidence_bundle(
                    seed,
                    code,
                    list(fallback_bundle.get("evidence") or scoped_web),
                    profile_raw,
                    info_raw,
                    status=reason_code,
                    identity=identity,
                    impact=impact,
                    path_specific_business=str(fallback_bundle.get("business_text") or "")
                    if web_fallback
                    else " ".join(business_values),
                    negative_evidence=["CNFinancial 未完成交叉验证；当前仅有两处独立 Web 证据。"]
                    if web_fallback
                    else [],
                    profile_log=profile_log,
                    info_log=info_log,
                )
            )

        if not accepted:
            continue
        impact_ids = _unique(str(item["item"]["seed"].get("impact_id") or "") for item in accepted)
        first_impact = impacts[impact_ids[0]]
        web_fallback_impacts = _unique(
            str(item["item"]["seed"].get("impact_id") or "")
            for item in accepted
            if item["web_fallback"]
        )
        business_by_impact = {
            str(item["item"]["seed"].get("impact_id") or ""): str(item["business_text"] or "")
            for item in accepted
        }
        verified_path_terms_by_impact = {
            str(item["item"]["seed"].get("impact_id") or ""): list(item.get("path_terms") or [])
            for item in accepted
            if item.get("path_terms")
        }
        all_web_evidence = _dedupe_evidence(
            [evidence for item in accepted for evidence in item.get("web_evidence") or []]
        )
        profile_evidence = normalize_mcp_evidence(
            profile_raw,
            query=code,
            server_name=CNFINANCIAL_SERVER,
            tool_name="get_company_profile",
        )
        info_evidence = normalize_mcp_evidence(
            info_raw,
            query=code,
            server_name=CNFINANCIAL_SERVER,
            tool_name="get_company_info",
        )
        candidate = {
            "company_name": str(accepted[0]["company_name"] or ""),
            "stock_code": code,
            "industry_segment": str(first_impact.get("chain_segment") or first_impact.get("industry") or ""),
            "chain_segment": str(first_impact.get("chain_segment") or first_impact.get("industry") or ""),
            "matched_business": " ".join(_unique(business_by_impact.values())),
            "business_evidence": " ".join(_unique(business_by_impact.values())),
            "business_evidence_by_impact": business_by_impact,
            "verified_path_terms_by_impact": verified_path_terms_by_impact,
            "business_keywords": _unique(
                keyword for impact_id in impact_ids for keyword in _company_keywords(impacts[impact_id])
            ),
            "source_name": "CNFinancial MCP" if not web_fallback_impacts else "independent Web evidence; CNFinancial cross-check incomplete",
            "source_url": str((all_web_evidence[0].get("source_url") if all_web_evidence else "") or ""),
            "data_date": str((all_web_evidence[0].get("published_date") if all_web_evidence else "") or "unknown"),
            "revenue_relevance": "unknown",
            "candidate_source_tool": "web_fallback" if web_fallback_impacts else "get_company_profile",
            "impact_ids": impact_ids,
            "seed_reasons": _unique(str(item["item"]["seed"].get("seed_reason") or "") for item in accepted),
            "identity_verified": True,
            "identity_verification": "web_fallback" if web_fallback_impacts else "cnfinancial",
            "web_fallback_verified": bool(web_fallback_impacts),
            "web_fallback_impacts": web_fallback_impacts,
            "confidence_cap": 0.55 if web_fallback_impacts else 0.92,
            "provenance": [
                {
                    "impact_id": str(item["item"]["seed"].get("impact_id") or ""),
                    "seed_id": str(item["item"]["seed"].get("seed_id") or ""),
                    "seed_reason": str(item["item"]["seed"].get("seed_reason") or ""),
                    "origin_channels": list(item["item"]["seed"].get("origin_channels") or []),
                    "tool": "web_fallback" if item["web_fallback"] else "get_company_profile",
                    "tool_call_id": str(profile_log.get("tool_call_id") or info_log.get("tool_call_id") or ""),
                    "source_type": "web_fallback_verified" if item["web_fallback"] else "web_seed_cnfinancial_verified",
                }
                for item in accepted
            ],
            "mcp_evidence": _dedupe_evidence([*profile_evidence, *info_evidence, *all_web_evidence]),
            "cnfinancial_raw": _payload_items(profile_raw)[0] if _payload_items(profile_raw) else {},
        }
        candidates.append(candidate)

    return _dedupe_and_merge_companies(candidates), audit, bundles


def _web_discovery_reason(status: str) -> str:
    if status == "empty":
        return "web_empty"
    if status == "unavailable":
        return "discovery_error"
    if status == "error":
        return "discovery_error"
    return "web_empty"


def _web_first_audit_base(
    seed: dict[str, Any],
    *,
    seed_id: str,
    proposed_code: str,
) -> dict[str, Any]:
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "seed_id": seed_id,
        "impact_id": str(seed.get("impact_id") or ""),
        "proposed_name": str(seed.get("proposed_name") or ""),
        "proposed_stock_code": proposed_code,
        "source": ",".join(str(value) for value in (seed.get("origin_channels") or [])) or "llm",
        "url": "",
        "date": str(seed.get("time") or ""),
        "tool_call_id": str(seed.get("tool_call_id") or ""),
        "cache_hit": False,
    }


def _cnfinancial_log_reason(logs: list[dict[str, Any]]) -> str:
    statuses = {str(item.get("status") or "") for item in logs}
    if "unavailable" in statuses or any(bool(item.get("circuit_open")) for item in logs):
        return "cnfinancial_unavailable"
    if "error" in statuses:
        return "cnfinancial_error"
    if "skipped" in statuses:
        return "cnfinancial_error"
    if "ok" in statuses:
        return "cnfinancial_ok"
    return "cnfinancial_empty"


def _web_identity_code(
    evidence: list[dict[str, Any]],
    proposed_name: str,
) -> tuple[str, bool]:
    name_key = _normalize_company_name(proposed_name)
    codes: list[str] = []
    for item in evidence:
        raw = item.get("raw_payload") if isinstance(item.get("raw_payload"), dict) else {}
        text = _web_evidence_item_text(item)
        explicit_name = str(_first_value(raw, COMPANY_NAME_KEYS) or "")
        name_matches = bool(
            name_key
            and (
                name_key in _normalize_company_name(explicit_name)
                or name_key in _normalize_company_name(text)
            )
        )
        if not name_matches:
            continue
        explicit_code = _normalize_stock_code(str(_first_value(raw, COMPANY_CODE_KEYS) or ""))
        if explicit_code and re.fullmatch(r"\d{6}", explicit_code):
            codes.append(explicit_code)
        codes.extend(re.findall(r"(?<!\d)(\d{6})(?!\d)", text))
    unique_codes = _unique(codes)
    return (unique_codes[0] if len(unique_codes) == 1 else "", len(unique_codes) > 1)


def _qualify_web_fallback(
    seed: dict[str, Any],
    code: str,
    impact: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    if _contains_non_current_listing_marker(_web_evidence_text(evidence)):
        return {"status": "rejected", "reason_code": "non_current_a_share_identity"}
    identity_code, conflict = _web_identity_code(evidence, str(seed.get("proposed_name") or ""))
    if conflict or (identity_code and identity_code != code):
        return {"status": "rejected", "reason_code": "identity_conflict"}

    selected: list[dict[str, Any]] = []
    urls: set[str] = set()
    domains: set[str] = set()
    sources: set[str] = set()
    texts: list[str] = []
    for item in evidence:
        normalized_url = _normalized_evidence_url(str(item.get("source_url") or ""))
        domain = _normalized_evidence_domain(normalized_url)
        source = _normalized_evidence_source(str(item.get("source_org") or ""), domain)
        text = _web_evidence_item_text(item)
        if not normalized_url or not domain or not source or not text:
            continue
        if normalized_url in urls or domain in domains or source in sources:
            continue
        if any(_content_is_duplicate(text, previous) for previous in texts):
            continue
        selected.append(item)
        urls.add(normalized_url)
        domains.add(domain)
        sources.add(source)
        texts.append(text)
        if len(selected) == 2:
            break

    if len(selected) < 2:
        return {"status": "unresolved", "reason_code": "web_fallback_insufficient_independent_sources"}
    combined = _web_evidence_text(selected)
    proposed_name = str(seed.get("proposed_name") or "")
    if _normalize_company_name(proposed_name) not in _normalize_company_name(combined) or code not in combined:
        return {"status": "unresolved", "reason_code": "web_fallback_identity_unverified"}
    path_terms = _web_path_specific_terms(impact, combined)
    if not path_terms:
        return {"status": "rejected", "reason_code": "business_rejected"}
    return {
        "status": "verified",
        "reason_code": "web_fallback",
        "evidence": selected,
        "business_text": combined,
        "path_terms": path_terms,
    }


def _company_evidence_bundle(
    seed: dict[str, Any],
    code: str,
    web_evidence: list[dict[str, Any]],
    profile_raw: Any,
    info_raw: Any,
    *,
    status: str,
    identity: dict[str, Any] | None = None,
    impact: dict[str, Any] | None = None,
    path_specific_business: str = "",
    negative_evidence: list[str] | None = None,
    profile_log: dict[str, Any] | None = None,
    info_log: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = identity or {}
    impact = impact or {}
    profile_log = profile_log or {}
    info_log = info_log or {}
    return {
        "impact_id": str(seed.get("impact_id") or ""),
        "seed_id": str(seed.get("seed_id") or ""),
        "company_name": str(seed.get("proposed_name") or ""),
        "stock_code": code,
        "status": status,
        "identity": {
            "company_name": str(identity.get("company_name") or seed.get("proposed_name") or ""),
            "stock_code": str(identity.get("stock_code") or code),
            "status": str(identity.get("status") or ""),
            "reason_code": str(identity.get("reason_code") or ""),
            "source_url": str(identity.get("source_url") or ""),
            "data_date": str(identity.get("data_date") or seed.get("time") or "unknown"),
        },
        "path": {
            "industry": str(impact.get("industry") or ""),
            "chain_segment": str(impact.get("chain_segment") or impact.get("industry") or ""),
            "business_variables": list(impact.get("business_variables") or []),
        },
        "path_specific_business": path_specific_business,
        "negative_evidence": list(negative_evidence or []),
        "data_date": str(identity.get("data_date") or seed.get("time") or "unknown"),
        "tool_status": {
            "get_company_info": {
                "status": str(info_log.get("status") or ""),
                "tool_call_id": str(info_log.get("tool_call_id") or ""),
                "cache_hit": bool(info_log.get("cache_hit")),
            },
            "get_company_profile": {
                "status": str(profile_log.get("status") or ""),
                "tool_call_id": str(profile_log.get("tool_call_id") or ""),
                "cache_hit": bool(profile_log.get("cache_hit")),
            },
        },
        "cnfinancial_profile": _payload_items(profile_raw),
        "cnfinancial_info": _payload_items(info_raw),
        "web_evidence": list(web_evidence),
    }


def _company_scoped_web_evidence(
    evidence: list[dict[str, Any]],
    *,
    names: list[str],
    code: str,
) -> list[dict[str, Any]]:
    normalized_names = {
        _normalize_company_name(value)
        for value in names
        if _normalize_company_name(value)
    }
    scoped: list[dict[str, Any]] = []
    for item in evidence:
        text = _web_evidence_item_text(item)
        normalized_text = _normalize_company_name(text)
        if not normalized_names or not any(name in normalized_text for name in normalized_names):
            continue
        explicit_codes = set(re.findall(r"(?<!\d)(\d{6})(?!\d)", text))
        raw = item.get("raw_payload") if isinstance(item.get("raw_payload"), dict) else {}
        explicit_code = _normalize_stock_code(str(_first_value(raw, COMPANY_CODE_KEYS) or ""))
        if explicit_code:
            explicit_codes.add(explicit_code)
        if explicit_codes and code not in explicit_codes:
            continue
        scoped.append(item)
    return _dedupe_evidence(scoped)


def _web_evidence_text(evidence: list[dict[str, Any]]) -> str:
    return " ".join(_web_evidence_item_text(item) for item in evidence if isinstance(item, dict))


def _web_evidence_item_text(item: dict[str, Any]) -> str:
    raw = item.get("raw_payload") if isinstance(item.get("raw_payload"), dict) else {}
    return " ".join(
        str(value or "")
        for value in (
            item.get("title"),
            item.get("summary"),
            raw.get("content"),
            raw.get("description"),
            raw.get("text"),
            _first_value(raw, COMPANY_NAME_KEYS),
            _first_value(raw, COMPANY_CODE_KEYS),
        )
    ).strip()


def _normalized_evidence_url(value: str) -> str:
    try:
        parts = urlsplit(value.strip())
    except ValueError:
        return ""
    host = (parts.hostname or "").lower().strip(".")
    if not host:
        return ""
    path = re.sub(r"/+", "/", parts.path or "/").rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower() or "https", host, path, "", ""))


def _normalized_evidence_domain(value: str) -> str:
    try:
        host = (urlsplit(value).hostname or "").lower().strip(".")
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _normalized_evidence_source(source_org: str, domain: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]", "", source_org).lower()
    return normalized or domain


def _content_is_duplicate(left: str, right: str) -> bool:
    left_key = re.sub(r"\W+", "", left).lower()
    right_key = re.sub(r"\W+", "", right).lower()
    if not left_key or not right_key:
        return False
    if left_key == right_key or left_key in right_key or right_key in left_key:
        return True
    left_grams = {left_key[index : index + 3] for index in range(max(len(left_key) - 2, 0))}
    right_grams = {right_key[index : index + 3] for index in range(max(len(right_key) - 2, 0))}
    if not left_grams or not right_grams:
        return False
    similarity = len(left_grams & right_grams) / max(len(left_grams | right_grams), 1)
    return similarity >= 0.82


def _web_path_specific_terms(impact: dict[str, Any], evidence_text: str) -> list[str]:
    compact_evidence = _normalize_company_name(evidence_text)
    generic = {
        "服务", "制造", "电力", "能源", "新能源", "企业", "行业", "产业", "公司", "业务", "产品", "设备", "技术",
    }
    values = [
        impact.get("industry"),
        impact.get("chain_segment"),
        *(impact.get("business_variables") or []),
        *(impact.get("affected_company_types") or []),
    ]
    terms: list[str] = []
    for value in values:
        compact = _normalize_company_name(str(value or ""))
        if not compact:
            continue
        candidates = [compact]
        for width in (6, 5, 4, 3):
            if len(compact) >= width:
                candidates.extend(compact[index : index + width] for index in range(len(compact) - width + 1))
        for candidate in candidates:
            if candidate in generic or len(candidate) < 3:
                continue
            if candidate in compact_evidence:
                terms.append(candidate)
    return _unique(terms)[:8]


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
    active_invoker: MCPToolInvoker = runtime_mcp_invoker(invoker) if invoker is not None else UnavailableMCPInvoker()
    log_entry = _new_tool_log(server_name=server_name, tool_name=tool_name, arguments=arguments)
    log_entry.update(log_context or {})
    is_company_stock_search = server_name == CNFINANCIAL_SERVER and tool_name == "search_stock"
    budget_scope = str(log_entry.get("impact_id") or "unscoped")
    budget_reserved = False
    if is_company_stock_search:
        query = str(arguments.get("keyword") or arguments.get("query") or "").strip()
        log_entry.update(
            {
                "impact_id": budget_scope,
                "query": query,
                "query_budget_limit": COMPANY_SEARCH_STOCK_MAX_CALLS_PER_IMPACT,
                "actual_execution": False,
            }
        )
        if not _is_specific_stock_search_term(query):
            log_entry.update(
                {
                    "status": "skipped",
                    "skip_reason": "invalid_query",
                    "error": "Rejected non-specific or descriptive company search query",
                    "duration_ms": 0.0,
                    "query_budget_used": _query_budget_used(active_invoker, server_name, tool_name, budget_scope),
                }
            )
            _append_tool_log(tool_logs, log_entry)
            _record_mcp_event(log_entry)
            return [], log_entry
        if isinstance(active_invoker, RuntimeMCPInvoker):
            allowed, used = active_invoker.reserve_query_budget(
                server_name,
                tool_name,
                budget_scope,
                limit=COMPANY_SEARCH_STOCK_MAX_CALLS_PER_IMPACT,
            )
            log_entry["query_budget_used"] = used
            if not allowed:
                log_entry.update(
                    {
                        "status": "skipped",
                        "skip_reason": "query_budget",
                        "error": "Per-impact company search query budget exhausted",
                        "duration_ms": 0.0,
                    }
                )
                _append_tool_log(tool_logs, log_entry)
                _record_mcp_event(log_entry)
                return [], log_entry
            budget_reserved = True
    if isinstance(active_invoker, RuntimeMCPInvoker):
        health, first_check = active_invoker.preflight(server_name)
        if first_check:
            _record_mcp_health_event(server_name, health)
    started = perf_counter()
    try:
        raw = active_invoker.invoke(server_name, tool_name, arguments)
        if is_company_stock_search:
            log_entry["actual_execution"] = True
        error_message = mcp_payload_error_message(raw, server_name=server_name, tool_name=tool_name)
        if error_message:
            log_entry.update(
                {
                    "status": "error",
                    "error": error_message,
                    "count": 0,
                    "duration_ms": _elapsed_ms(started),
                    **_runtime_call_metadata(active_invoker),
                }
            )
            _append_tool_log(tool_logs, log_entry)
            _record_invoker_error(active_invoker, error_message)
            _record_mcp_event(log_entry)
            return [], log_entry
        count = _payload_count(raw)
        log_entry.update(
            {
                "status": "ok" if count else "empty",
                "count": count,
                "error": "",
                "duration_ms": _elapsed_ms(started),
                **_runtime_call_metadata(active_invoker),
            }
        )
        _append_tool_log(tool_logs, log_entry)
        _record_mcp_event(log_entry)
        return raw, log_entry
    except MCPToolUnavailable as exc:
        runtime_metadata = _runtime_call_metadata(active_invoker)
        circuit_skipped = bool(runtime_metadata.get("skipped"))
        if budget_reserved and circuit_skipped and isinstance(active_invoker, RuntimeMCPInvoker):
            log_entry["query_budget_used"] = active_invoker.release_query_budget(
                server_name, tool_name, budget_scope
            )
        log_entry.update(
            {
                "status": "unavailable",
                "error": str(exc),
                "count": 0,
                "duration_ms": _elapsed_ms(started),
                **runtime_metadata,
            }
        )
        if is_company_stock_search:
            log_entry["actual_execution"] = bool(invoker is not None and not circuit_skipped)
        _append_tool_log(tool_logs, log_entry)
        _record_invoker_error(active_invoker, str(exc))
        _record_mcp_event(log_entry)
        return [], log_entry
    except Exception as exc:
        runtime_metadata = _runtime_call_metadata(active_invoker)
        circuit_skipped = bool(runtime_metadata.get("skipped"))
        if budget_reserved and circuit_skipped and isinstance(active_invoker, RuntimeMCPInvoker):
            log_entry["query_budget_used"] = active_invoker.release_query_budget(
                server_name, tool_name, budget_scope
            )
        log_entry.update(
            {
                "status": "error",
                "error": str(exc),
                "count": 0,
                "duration_ms": _elapsed_ms(started),
                **runtime_metadata,
            }
        )
        if is_company_stock_search:
            log_entry["actual_execution"] = bool(invoker is not None and not circuit_skipped)
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


def _record_mcp_health_event(server_name: str, health: dict[str, Any]) -> None:
    try:
        from policychain.observability import record_event

        record_event(
            "mcp.health",
            stage="mcp",
            status=str(health.get("status") or "unknown"),
            server_name=server_name,
            check="run_preflight",
            circuit_open=bool(health.get("circuit_open")),
            circuit_reason=str(health.get("circuit_reason") or ""),
        )
    except Exception:
        return


def _runtime_call_metadata(invoker: MCPToolInvoker) -> dict[str, Any]:
    metadata_reader = getattr(invoker, "call_metadata", None)
    if not callable(metadata_reader):
        return {}
    metadata = dict(metadata_reader())
    return {
        key: metadata.get(key)
        for key in (
            "cache_hit",
            "skipped",
            "circuit_open",
            "circuit_scope",
            "failure_count",
            "server_status",
            "tool_status",
        )
        if key in metadata
    }


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
    return _unique(terms)[:4]


def _candidate_stock_search_terms(impact: dict[str, Any]) -> list[str]:
    source_values = [
        impact.get("chain_segment"),
        impact.get("industry"),
        *(impact.get("affected_company_types") or []),
    ]
    terms: list[str] = []
    for value in source_values:
        raw = re.sub(r"\s+", "", str(value or "")).strip("，。；;、/|：:")
        if not raw or len(raw) > 24:
            continue
        reduced = raw
        for suffix in GENERIC_TERM_SUFFIXES:
            if reduced.endswith(suffix) and len(reduced) > len(suffix) + 1:
                reduced = reduced[: -len(suffix)]
                break
        if reduced != raw:
            terms.append(reduced)
        terms.append(raw)
    return _unique([term for term in terms if _is_specific_stock_search_term(term)])[:4]


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


def _is_specific_stock_search_term(value: str) -> bool:
    term = re.sub(r"\s+", "", str(value or "")).strip("，。；;、/|：:")
    if not (2 <= len(term) <= 24):
        return False
    if term in GENERIC_STOCK_SEARCH_TERMS:
        return False
    if any(marker in term for marker in INVALID_STOCK_SEARCH_MARKERS):
        return False
    if re.search(r"[\r\n。！？!?；;：:,，、/|]", term):
        return False
    if any(action in term for action in INVALID_STOCK_SEARCH_ACTIONS):
        return False
    if any(metric in term for metric in INVALID_STOCK_SEARCH_METRIC_TERMS):
        return False
    if term.endswith("环节"):
        return False
    if any(term.endswith(suffix) for suffix in GENERIC_TERM_SUFFIXES):
        return False
    return True


def _query_budget_used(
    invoker: MCPToolInvoker,
    server_name: str,
    tool_name: str,
    scope_id: str,
) -> int:
    if not isinstance(invoker, RuntimeMCPInvoker):
        return 0
    snapshot = invoker.status_snapshot().get("query_budgets") or {}
    return int(snapshot.get(f"{server_name}.{tool_name}.{scope_id}") or 0)


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
    recall_logs: list[dict[str, Any]],
    query_terms: list[str],
) -> None:
    log_entry = _new_tool_log(
        server_name="policychain",
        tool_name="company_candidate_pipeline",
        arguments={"impact_id": impact_id},
    )
    statuses = [str(item.get("status") or "") for item in recall_logs]
    search_logs = [item for item in recall_logs if str(item.get("tool_name") or "") == "search_stock"]
    executed_queries = [
        str(item.get("query") or (item.get("arguments") or {}).get("keyword") or "")
        for item in search_logs
        if item.get("actual_execution") is True
    ]
    skipped_queries = [
        {
            "query": str(item.get("query") or (item.get("arguments") or {}).get("keyword") or ""),
            "skip_reason": str(item.get("skip_reason") or ""),
        }
        for item in search_logs
        if str(item.get("status") or "") == "skipped"
    ]
    errors = _unique([str(item.get("error") or "") for item in recall_logs if item.get("error")])
    if kept_count:
        status = "ok"
    elif "error" in statuses:
        status = "error"
    elif "unavailable" in statuses:
        status = "unavailable"
    else:
        status = "empty"
    channel_statuses: dict[str, list[str]] = {}
    for item in recall_logs:
        tool_name = str(item.get("tool_name") or "unknown")
        channel_statuses.setdefault(tool_name, [])
        item_status = str(item.get("status") or "unknown")
        if item_status not in channel_statuses[tool_name]:
            channel_statuses[tool_name].append(item_status)
    log_entry.update(
        {
            "impact_id": impact_id,
            "status": status,
            "count": kept_count,
            "raw_count": raw_count,
            "dedup_count": dedup_count,
            "truncated_count": max(dedup_count - kept_count, 0),
            "error": " | ".join(errors[:3]),
            "query_terms": executed_queries,
            "query_count": len(executed_queries),
            "requested_query_terms": list(query_terms),
            "skipped_queries": skipped_queries,
            "skipped_query_count": len(skipped_queries),
            "channel_statuses": channel_statuses,
            "partial_failure_count": sum(1 for value in statuses if value in {"error", "unavailable"}),
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


def candidate_retrieval_statuses(tool_logs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return per-impact recall semantics without treating failures as true empties."""

    output: dict[str, dict[str, Any]] = {}
    for item in tool_logs:
        if str(item.get("tool_name") or "") != "company_candidate_pipeline":
            continue
        impact_id = str(item.get("impact_id") or "")
        if not impact_id:
            continue
        output[impact_id] = {
            "status": str(item.get("status") or "empty"),
            "error": str(item.get("error") or ""),
            "query_terms": list(item.get("query_terms") or []),
            "query_count": int(item.get("query_count") or 0),
            "requested_query_terms": list(item.get("requested_query_terms") or []),
            "skipped_queries": list(item.get("skipped_queries") or []),
            "skipped_query_count": int(item.get("skipped_query_count") or 0),
            "channel_statuses": dict(item.get("channel_statuses") or {}),
            "partial_failure_count": int(item.get("partial_failure_count") or 0),
        }
    for impact_id, status in output.items():
        search_logs = [
            item
            for item in tool_logs
            if str(item.get("tool_name") or "") == "search_stock"
            and str(item.get("impact_id") or "") == impact_id
        ]
        executed_queries = [
            str(item.get("query") or (item.get("arguments") or {}).get("keyword") or "")
            for item in search_logs
            if item.get("actual_execution") is True
        ]
        skipped_queries = [
            {
                "query": str(item.get("query") or (item.get("arguments") or {}).get("keyword") or ""),
                "skip_reason": str(item.get("skip_reason") or ""),
            }
            for item in search_logs
            if str(item.get("status") or "") == "skipped"
        ]
        status["query_terms"] = executed_queries
        status["query_count"] = len(executed_queries)
        status["skipped_queries"] = skipped_queries
        status["skipped_query_count"] = len(skipped_queries)
    return output


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
