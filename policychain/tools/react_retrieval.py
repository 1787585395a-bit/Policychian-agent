from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable, Protocol

from policychain.mcp import MCPToolInvoker


class ReActLLMClient(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return a JSON decision for the next retrieval action."""


@dataclass(frozen=True)
class ReActTool:
    name: str
    description: str
    run: Callable[[dict[str, Any]], Any]


@dataclass
class ReActRun:
    evidence: list[dict[str, Any]] = field(default_factory=list)
    traces: list[dict[str, Any]] = field(default_factory=list)


class ReActRetrievalError(RuntimeError):
    """Raised when a ReAct retrieval loop cannot execute a planned action."""


def run_react_retrieval(
    goal: str,
    tools: list[ReActTool],
    llm_client: ReActLLMClient,
    max_steps: int = 3,
) -> ReActRun:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not goal.strip():
        return ReActRun()

    tool_map = {tool.name: tool for tool in tools}
    traces: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    observations: list[str] = []

    for step_index in range(1, max_steps + 1):
        decision = _next_decision(goal, tools, observations, llm_client)
        action = str(decision.get("action") or decision.get("tool") or "").strip()
        if action.lower() in {"finish", "stop", "final"}:
            traces.append(
                {
                    "step": step_index,
                    "action": "finish",
                    "thought": str(decision.get("thought") or ""),
                    "message": str(decision.get("message") or decision.get("reason") or ""),
                }
            )
            break

        if action not in tool_map:
            trace = {
                "step": step_index,
                "action": action or "unknown",
                "error": f"Unknown ReAct tool: {action or 'empty'}",
                "available_tools": sorted(tool_map),
            }
            traces.append(trace)
            break

        arguments = decision.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {"query": str(arguments)}

        try:
            raw_observation = tool_map[action].run(arguments)
            tool_evidence = _evidence_items(raw_observation)
            evidence.extend(tool_evidence)
            observation_summary = _observation_summary(raw_observation)
            observations.append(f"{action}: {observation_summary}")
            traces.append(
                {
                    "step": step_index,
                    "action": action,
                    "thought": str(decision.get("thought") or ""),
                    "arguments": arguments,
                    "observation": observation_summary,
                    "evidence_count": len(tool_evidence),
                }
            )
        except Exception as exc:
            traces.append(
                {
                    "step": step_index,
                    "action": action,
                    "thought": str(decision.get("thought") or ""),
                    "arguments": arguments,
                    "error": str(exc),
                }
            )
            break

    return ReActRun(evidence=_dedupe_evidence(evidence), traces=traces)


def build_langchain_tools(tools: list[ReActTool]) -> list[Any]:
    """Create LangChain Tool wrappers when LangChain is installed.

    The project keeps this optional so the default offline workflow and tests do
    not fail before users install optional LangChain dependencies.
    """

    try:
        from langchain_core.tools import Tool
    except ImportError:
        return []

    wrapped = []
    for tool in tools:
        wrapped.append(
            Tool.from_function(
                name=tool.name,
                description=tool.description,
                func=_langchain_tool_func(tool),
            )
        )
    return wrapped


def run_policy_react_search(
    query: str,
    invoker: MCPToolInvoker | None,
    llm_client: ReActLLMClient,
    top_k: int = 3,
    max_steps: int = 3,
) -> ReActRun:
    from policychain.tools.mcp_tools import POLICY_SOURCE_PRIORITY, fetch_web_content, search_web

    tools = [
        ReActTool(
            name="web.search",
            description="Search external web evidence for policy comparison, official interpretation, or implementation details.",
            run=lambda args: search_web(
                str(args.get("query") or query),
                source_priority=POLICY_SOURCE_PRIORITY,
                top_k=int(args.get("top_k") or top_k),
                invoker=invoker,
            ),
        ),
        ReActTool(
            name="web.fetch",
            description="Fetch a specific URL returned by web search.",
            run=lambda args: fetch_web_content(str(args.get("url") or ""), invoker=invoker),
        ),
    ]
    return run_react_retrieval(
        goal=f"Find comparable policy evidence and official context for: {query}",
        tools=tools,
        llm_client=llm_client,
        max_steps=max_steps,
    )


def run_impact_react_search(
    query: str,
    invoker: MCPToolInvoker | None,
    llm_client: ReActLLMClient,
    top_k: int = 3,
    max_steps: int = 3,
) -> ReActRun:
    from policychain.tools.mcp_tools import (
        CNFINANCIAL_SERVER,
        INDUSTRY_SOURCE_PRIORITY,
        normalize_mcp_evidence,
        search_web,
        _invoke_or_empty,
    )

    def cnfinancial(tool_name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        raw = _invoke_or_empty(invoker, CNFINANCIAL_SERVER, tool_name, arguments)
        return normalize_mcp_evidence(raw, query=query, server_name=CNFINANCIAL_SERVER, tool_name=tool_name)

    tools = [
        ReActTool("cnfinancial.get_industry_list", "List A-share industries available through CNFinancial.", lambda args: cnfinancial("get_industry_list", {})),
        ReActTool("cnfinancial.get_concept_list", "List A-share concepts available through CNFinancial.", lambda args: cnfinancial("get_concept_list", {})),
        ReActTool("cnfinancial.search_news", "Search industry news through CNFinancial.", lambda args: cnfinancial("search_news", {"keyword": str(args.get("query") or query), "num_results": int(args.get("top_k") or top_k)})),
        ReActTool("web.search", "Search government, statistics, association, and authoritative industry evidence.", lambda args: search_web(str(args.get("query") or query), source_priority=INDUSTRY_SOURCE_PRIORITY, top_k=int(args.get("top_k") or top_k), invoker=invoker)),
    ]
    required_run = _run_required_tools(
        goal=query,
        tools=tools,
        required_actions=[
            ("cnfinancial.get_industry_list", {}),
            ("cnfinancial.get_concept_list", {}),
            ("cnfinancial.search_news", {"query": query, "top_k": top_k}),
        ],
    )
    planned_run = run_react_retrieval(
        goal=f"Find industry data and transmission evidence for: {query}",
        tools=tools,
        llm_client=llm_client,
        max_steps=max_steps,
    )
    return _merge_react_runs(required_run, planned_run)


def run_company_react_search(
    query: str,
    invoker: MCPToolInvoker | None,
    llm_client: ReActLLMClient,
    top_k: int = 3,
    max_steps: int = 3,
) -> ReActRun:
    from policychain.tools.mcp_tools import (
        CNFINANCIAL_SERVER,
        COMPANY_SOURCE_PRIORITY,
        normalize_mcp_evidence,
        search_web,
        _invoke_or_empty,
        _load_cnfinancial_sector_catalogs,
        _sector_search_terms,
        _select_cnfinancial_sectors,
        _selected_sector_names,
    )

    industry_catalog, concept_catalog = _load_cnfinancial_sector_catalogs(invoker)
    sector_selection = _select_cnfinancial_sectors(
        terms=[query],
        industry_impacts=[],
        industry_catalog=industry_catalog,
        concept_catalog=concept_catalog,
    )
    legal_industries = _selected_sector_names(sector_selection, "selected_industries")
    search_terms = _sector_search_terms(sector_selection, [query])

    def cnfinancial(tool_name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        raw = _invoke_or_empty(invoker, CNFINANCIAL_SERVER, tool_name, arguments)
        return normalize_mcp_evidence(raw, query=query, server_name=CNFINANCIAL_SERVER, tool_name=tool_name)

    def industry_stocks(arguments: dict[str, Any]) -> list[dict[str, Any]]:
        industry = str(arguments.get("industry") or arguments.get("query") or "")
        if industry not in legal_industries:
            return []
        return cnfinancial("get_industry_stocks", {"industry": industry})

    tools = [
        ReActTool("cnfinancial.search_stock", "Search A-share candidate companies by business or product keyword.", lambda args: cnfinancial("search_stock", {"keyword": str(args.get("query") or query)})),
        ReActTool("cnfinancial.get_industry_stocks", "Fetch A-share companies in a legal CNFinancial industry board.", industry_stocks),
        ReActTool("web.search", "Search exchange filings, company announcements, and official company evidence.", lambda args: search_web(str(args.get("query") or query), source_priority=COMPANY_SOURCE_PRIORITY, top_k=int(args.get("top_k") or top_k), invoker=invoker)),
    ]
    required_actions: list[tuple[str, dict[str, Any]]] = [
        ("cnfinancial.search_stock", {"query": (search_terms[0] if search_terms else query)}),
    ]
    if legal_industries:
        required_actions.insert(0, ("cnfinancial.get_industry_stocks", {"industry": legal_industries[0]}))
    required_run = _run_required_tools(
        goal=query,
        tools=tools,
        required_actions=required_actions,
    )
    planned_run = run_react_retrieval(
        goal=f"Find A-share company business evidence for: {query}",
        tools=tools,
        llm_client=llm_client,
        max_steps=max_steps,
    )
    return _merge_react_runs(required_run, planned_run)


def _run_required_tools(
    goal: str,
    tools: list[ReActTool],
    required_actions: list[tuple[str, dict[str, Any]]],
) -> ReActRun:
    tool_map = {tool.name: tool for tool in tools}
    traces: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for index, (action, arguments) in enumerate(required_actions, start=1):
        tool = tool_map.get(action)
        if tool is None:
            traces.append(
                {
                    "step": f"required-{index}",
                    "action": action,
                    "arguments": arguments,
                    "error": f"Required ReAct tool is unavailable: {action}",
                }
            )
            continue
        try:
            raw_observation = tool.run(arguments)
            tool_evidence = _evidence_items(raw_observation)
            evidence.extend(tool_evidence)
            traces.append(
                {
                    "step": f"required-{index}",
                    "action": action,
                    "thought": "required CNFinancial-first evidence probe",
                    "arguments": arguments,
                    "observation": _observation_summary(raw_observation),
                    "evidence_count": len(tool_evidence),
                }
            )
        except Exception as exc:
            traces.append(
                {
                    "step": f"required-{index}",
                    "action": action,
                    "thought": "required CNFinancial-first evidence probe",
                    "arguments": arguments,
                    "error": str(exc),
                }
            )
    return ReActRun(evidence=_dedupe_evidence(evidence), traces=traces)


def _merge_react_runs(first: ReActRun, second: ReActRun) -> ReActRun:
    return ReActRun(
        evidence=_dedupe_evidence([*first.evidence, *second.evidence]),
        traces=[*first.traces, *second.traces],
    )


def _next_decision(
    goal: str,
    tools: list[ReActTool],
    observations: list[str],
    llm_client: ReActLLMClient,
) -> dict[str, Any]:
    system_prompt = (
        "You are a retrieval planner for PolicyChain. "
        "Choose one tool call at a time, observe the result, then refine the next query if needed. "
        "Return only JSON with keys: thought, action, arguments. "
        "Use action='finish' when enough evidence has been collected."
    )
    tool_text = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
    user_prompt = (
        f"Goal:\n{goal}\n\n"
        f"Available tools:\n{tool_text}\n\n"
        f"Previous observations:\n{json.dumps(observations, ensure_ascii=False, indent=2)}\n\n"
        "Return the next retrieval decision as JSON."
    )
    raw = llm_client.generate(system_prompt, user_prompt)
    return _parse_json_object(raw)


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ReActRetrievalError("ReAct planner did not return valid JSON") from exc
    if not isinstance(payload, dict):
        raise ReActRetrievalError("ReAct planner response must be a JSON object")
    return payload


def _langchain_tool_func(tool: ReActTool) -> Callable[[str], str]:
    def _run(arguments_json: str) -> str:
        try:
            arguments = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError:
            arguments = {"query": arguments_json}
        result = tool.run(arguments if isinstance(arguments, dict) else {"query": str(arguments)})
        return json.dumps(result, ensure_ascii=False, default=str)

    return _run


def _evidence_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("results", "data", "items", "reports", "announcements", "stocks", "companies"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [value]
    return []


def _observation_summary(value: Any, max_chars: int = 500) -> str:
    items = _evidence_items(value)
    if not items:
        text = str(value)
    else:
        parts = []
        for item in items[:3]:
            parts.append(
                " / ".join(
                    str(item.get(key) or "")
                    for key in ("title", "name", "source_org", "summary", "content")
                    if item.get(key)
                )
            )
        text = "; ".join(part for part in parts if part) or f"{len(items)} item(s)"
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[: max_chars - 1] + "..."


def _dedupe_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for item in items:
        key = (
            str(item.get("server_name") or item.get("source_name") or ""),
            str(item.get("tool_name") or ""),
            str(item.get("source_url") or item.get("title") or item.get("company_name") or item.get("query") or ""),
        )
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output
