from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
import re
from typing import Any

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
    for topic in POLICY_WEB_TOPICS:
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
) -> dict[str, list[dict[str, Any]]]:
    industry_terms = _industry_terms(policy_analysis, industry_impacts)
    financial: list[dict[str, Any]] = []
    web: list[dict[str, Any]] = []
    tool_logs: list[dict[str, Any]] = []

    for term in industry_terms:
        for tool_name in CNFINANCIAL_IMPACT_TOOLS:
            for query_term in _cnfinancial_impact_query_terms(tool_name, term):
                raw = _invoke_or_empty(
                    invoker=invoker,
                    server_name=CNFINANCIAL_SERVER,
                    tool_name=tool_name,
                    arguments=_cnfinancial_impact_arguments(tool_name, query_term),
                    tool_logs=tool_logs,
                )
                financial.extend(
                    normalize_mcp_evidence(
                        raw,
                        query=query_term or term or tool_name,
                        server_name=CNFINANCIAL_SERVER,
                        tool_name=tool_name,
                    )
                )
        web.extend(
            search_web(
                query=f"{term} 行业数据 国家统计局 行业协会 产量 销量 价格 库存 产能 技术路线",
                source_priority=INDUSTRY_SOURCE_PRIORITY,
                top_k=top_k,
                invoker=invoker,
                tool_logs=tool_logs,
            )
        )

    return {
        "cnfinancial": _dedupe_evidence(financial),
        "web": _dedupe_evidence(web),
        "tool_logs": tool_logs,
    }


def collect_company_candidates(
    industry_impacts: list[dict[str, Any]],
    invoker: MCPToolInvoker | None = None,
    top_k_per_industry: int = 3,
    tool_logs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for impact in industry_impacts:
        for industry in _candidate_industry_terms(impact):
            raw = _invoke_or_empty(
                invoker=invoker,
                server_name=CNFINANCIAL_SERVER,
                tool_name="get_industry_stocks",
                arguments={"industry": industry},
                tool_logs=tool_logs,
            )
            for item in _payload_items(raw)[:top_k_per_industry]:
                candidate = _normalize_company_candidate(item, impact, industry_segment=industry)
                if not candidate.get("company_name"):
                    continue
                candidate["mcp_evidence"] = normalize_mcp_evidence(
                    item,
                    query=industry,
                    server_name=CNFINANCIAL_SERVER,
                    tool_name="get_industry_stocks",
                )
                _enrich_company_candidate(candidate, invoker=invoker, tool_logs=tool_logs)
                candidates.append(candidate)

        if len(_dedupe_companies(candidates)) >= top_k_per_industry:
            continue

        for term in _candidate_stock_search_terms(impact):
            raw = _invoke_or_empty(
                invoker=invoker,
                server_name=CNFINANCIAL_SERVER,
                tool_name="search_stock",
                arguments={"keyword": term},
                tool_logs=tool_logs,
            )
            for item in _payload_items(raw)[:top_k_per_industry]:
                candidate = _normalize_company_candidate(item, impact)
                if not candidate.get("company_name"):
                    continue
                candidate["mcp_evidence"] = normalize_mcp_evidence(
                    item,
                    query=term,
                    server_name=CNFINANCIAL_SERVER,
                    tool_name="search_stock",
                )
                _enrich_company_candidate(candidate, invoker=invoker, tool_logs=tool_logs)
                candidates.append(candidate)
            if len(_dedupe_companies(candidates)) >= top_k_per_industry:
                break
    return _dedupe_companies(candidates)


def collect_impact_research(
    policy_analysis: dict[str, Any],
    industry_impacts: list[dict[str, Any]],
    invoker: MCPToolInvoker | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    industry_terms = _industry_terms(policy_analysis, industry_impacts)
    financial: list[dict[str, Any]] = []
    web: list[dict[str, Any]] = []
    tool_logs: list[dict[str, Any]] = []

    industry_catalog, concept_catalog = _load_cnfinancial_sector_catalogs(invoker, tool_logs)
    financial.extend(
        normalize_mcp_evidence(
            industry_catalog,
            query="get_industry_list",
            server_name=CNFINANCIAL_SERVER,
            tool_name="get_industry_list",
        )
    )
    financial.extend(
        normalize_mcp_evidence(
            concept_catalog,
            query="get_concept_list",
            server_name=CNFINANCIAL_SERVER,
            tool_name="get_concept_list",
        )
    )

    sector_selection = _select_cnfinancial_sectors(
        terms=industry_terms,
        industry_impacts=industry_impacts,
        industry_catalog=industry_catalog,
        concept_catalog=concept_catalog,
    )
    _append_sector_selection_log(tool_logs, sector_selection)
    financial.append(_sector_selection_evidence(sector_selection))

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

    for industry in _selected_sector_names(sector_selection, "selected_industries"):
        for tool_name in ("get_industry_stocks", "get_industry_pe"):
            raw = _invoke_or_empty(
                invoker=invoker,
                server_name=CNFINANCIAL_SERVER,
                tool_name=tool_name,
                arguments={"industry": industry},
                tool_logs=tool_logs,
            )
            financial.extend(
                normalize_mcp_evidence(
                    raw,
                    query=industry,
                    server_name=CNFINANCIAL_SERVER,
                    tool_name=tool_name,
                )
            )

    for term in _sector_search_terms(sector_selection, industry_terms):
        raw = _invoke_or_empty(
            invoker=invoker,
            server_name=CNFINANCIAL_SERVER,
            tool_name="search_news",
            arguments={"keyword": term, "num_results": 5},
            tool_logs=tool_logs,
        )
        financial.extend(
            normalize_mcp_evidence(
                raw,
                query=term,
                server_name=CNFINANCIAL_SERVER,
                tool_name="search_news",
            )
        )

    for term in industry_terms:
        web.extend(
            search_web(
                query=f"{term} industry data statistics association production sales price inventory capacity technology route",
                source_priority=INDUSTRY_SOURCE_PRIORITY,
                top_k=top_k,
                invoker=invoker,
                tool_logs=tool_logs,
            )
        )

    return {
        "cnfinancial": _dedupe_evidence(financial),
        "web": _dedupe_evidence(web),
        "tool_logs": tool_logs,
        "sector_selection": sector_selection,
    }


def collect_company_candidates(
    industry_impacts: list[dict[str, Any]],
    invoker: MCPToolInvoker | None = None,
    top_k_per_industry: int = 3,
    tool_logs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    industry_terms = _industry_terms({}, industry_impacts)
    industry_catalog, concept_catalog = _load_cnfinancial_sector_catalogs(invoker, tool_logs)
    sector_selection = _select_cnfinancial_sectors(
        terms=industry_terms,
        industry_impacts=industry_impacts,
        industry_catalog=industry_catalog,
        concept_catalog=concept_catalog,
    )
    _append_sector_selection_log(tool_logs, sector_selection)
    legal_industries = _selected_sector_names(sector_selection, "selected_industries")
    selected_concepts = _selected_sector_names(sector_selection, "selected_concepts")
    search_terms = _sector_search_terms(sector_selection, industry_terms)

    for impact in industry_impacts:
        for industry in legal_industries:
            raw = _invoke_or_empty(
                invoker=invoker,
                server_name=CNFINANCIAL_SERVER,
                tool_name="get_industry_stocks",
                arguments={"industry": industry},
                tool_logs=tool_logs,
            )
            for item in _payload_items(raw)[:top_k_per_industry]:
                candidate = _normalize_company_candidate(item, impact, industry_segment=industry)
                if not candidate.get("company_name"):
                    continue
                candidate["candidate_source_tool"] = "get_industry_stocks"
                candidate["candidate_query"] = industry
                candidate["selected_industries"] = legal_industries
                candidate["selected_concepts"] = selected_concepts
                candidate["mcp_evidence"] = normalize_mcp_evidence(
                    item,
                    query=industry,
                    server_name=CNFINANCIAL_SERVER,
                    tool_name="get_industry_stocks",
                )
                _enrich_company_candidate(candidate, invoker=invoker, tool_logs=tool_logs)
                candidates.append(candidate)

        if len(_dedupe_companies(candidates)) >= top_k_per_industry:
            continue

        for term in _unique([*search_terms, *_candidate_stock_search_terms(impact)]):
            raw = _invoke_or_empty(
                invoker=invoker,
                server_name=CNFINANCIAL_SERVER,
                tool_name="search_stock",
                arguments={"keyword": term},
                tool_logs=tool_logs,
            )
            for item in _payload_items(raw)[:top_k_per_industry]:
                candidate = _normalize_company_candidate(item, impact)
                if not candidate.get("company_name"):
                    continue
                candidate["candidate_source_tool"] = "search_stock"
                candidate["candidate_query"] = term
                candidate["selected_industries"] = legal_industries
                candidate["selected_concepts"] = selected_concepts
                candidate["mcp_evidence"] = normalize_mcp_evidence(
                    item,
                    query=term,
                    server_name=CNFINANCIAL_SERVER,
                    tool_name="search_stock",
                )
                _enrich_company_candidate(candidate, invoker=invoker, tool_logs=tool_logs)
                candidates.append(candidate)
            if len(_dedupe_companies(candidates)) >= top_k_per_industry:
                break
    return _dedupe_companies(candidates)


def collect_company_web_evidence(
    company_records: list[dict[str, Any]],
    invoker: MCPToolInvoker | None = None,
    top_k: int = 2,
    tool_logs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
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
) -> Any:
    active_invoker = invoker or UnavailableMCPInvoker()
    log_entry = _new_tool_log(server_name=server_name, tool_name=tool_name, arguments=arguments)
    try:
        raw = active_invoker.invoke(server_name, tool_name, arguments)
        error_message = mcp_payload_error_message(raw, server_name=server_name, tool_name=tool_name)
        if error_message:
            log_entry.update({"status": "error", "error": error_message, "count": 0})
            _append_tool_log(tool_logs, log_entry)
            _record_invoker_error(active_invoker, error_message)
            return []
        log_entry.update({"status": "ok", "count": _payload_count(raw), "error": ""})
        _append_tool_log(tool_logs, log_entry)
        return raw
    except (MCPToolError, MCPToolUnavailable) as exc:
        log_entry.update({"status": "error", "error": str(exc), "count": 0})
        _append_tool_log(tool_logs, log_entry)
        _record_invoker_error(active_invoker, str(exc))
        return []


def _new_tool_log(server_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "server_name": server_name,
        "tool_name": tool_name,
        "arguments": dict(arguments),
        "status": "pending",
        "count": 0,
        "error": "",
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


def _append_sector_selection_log(tool_logs: list[dict[str, Any]] | None, selection: dict[str, Any]) -> None:
    log_entry = _new_tool_log(
        server_name=CNFINANCIAL_SERVER,
        tool_name="select_sectors",
        arguments={"keywords": selection.get("keywords") or []},
    )
    selected_count = len(selection.get("selected_industries") or []) + len(selection.get("selected_concepts") or [])
    log_entry.update(
        {
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
        "source_priority": [],
        "raw_payload": selection,
    }


def _selected_sector_names(selection: dict[str, Any], key: str) -> list[str]:
    return [str(item.get("name") or "") for item in selection.get(key) or [] if item.get("name")]


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


def _normalize_company_candidate(
    item: dict[str, Any],
    impact: dict[str, Any],
    industry_segment: str | None = None,
) -> dict[str, Any]:
    name = _first_value(item, ("company_name", "name", "stock_name", "证券简称", "股票简称"))
    code = _first_value(item, ("stock_code", "code", "symbol", "证券代码", "股票代码"))
    industry = str(impact.get("chain_segment") or impact.get("industry") or item.get("industry") or "")
    return {
        "company_name": str(name or ""),
        "stock_code": str(code or ""),
        "industry_segment": industry,
        "chain_segment": str(impact.get("chain_segment") or industry),
        "matched_business": str(_first_value(item, ("matched_business", "main_business", "主营业务", "business")) or ""),
        "business_keywords": _company_keywords(impact),
        "source_name": "CNFinancial MCP",
        "source_url": str(_first_value(item, ("source_url", "url", "link")) or ""),
        "business_evidence": str(_first_value(item, ("business_evidence", "summary", "description", "主营业务")) or ""),
        "data_date": str(_first_value(item, ("data_date", "date", "publish_date")) or ""),
        "revenue_relevance": "unknown",
        "cnfinancial_raw": item,
    }


def _enrich_company_candidate(
    candidate: dict[str, Any],
    invoker: MCPToolInvoker | None,
    tool_logs: list[dict[str, Any]] | None = None,
) -> None:
    stock_code = str(candidate.get("stock_code") or "")
    if not stock_code:
        return
    evidence: list[dict[str, Any]] = []
    for tool_name in CNFINANCIAL_COMPANY_TOOLS:
        if tool_name == "search_stock":
            continue
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


def _merge_company_profile(candidate: dict[str, Any], raw: Any) -> None:
    for item in _payload_items(raw):
        business = _first_value(item, ("main_business", "主营业务", "business", "description", "summary"))
        revenue = _first_value(item, ("revenue_relevance", "revenue_ratio", "收入占比", "ratio"))
        if business and not candidate.get("matched_business"):
            candidate["matched_business"] = str(business)
        if business and not candidate.get("business_evidence"):
            candidate["business_evidence"] = str(business)
        if revenue:
            candidate["revenue_relevance"] = str(revenue)


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
    seen: set[tuple[str, str]] = set()
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (str(candidate.get("company_name") or ""), str(candidate.get("stock_code") or ""))
        if key not in seen:
            seen.add(key)
            output.append(candidate)
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
