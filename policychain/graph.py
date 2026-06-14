from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from policychain.agents import (
    run_company_matcher,
    run_impact_analyst,
    run_llm_company_matcher,
    run_llm_impact_analyst,
    run_llm_policy_analyst,
    run_policy_analyst,
)
from policychain.llm import LLMClient, create_llm_client
from policychain.mcp import MCPToolInvoker
from policychain.agents.report_writer import write_research_report
from policychain.state import PolicyResearchState
from policychain.storage.sqlite_store import SQLitePolicyStore


ProgressCallback = Callable[[int, str, str], None]


def run_policy_research_workflow(
    user_query: str,
    store: SQLitePolicyStore,
    mcp_invoker: MCPToolInvoker | None = None,
    use_annual_reports: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> PolicyResearchState:
    state = PolicyResearchState(user_query=user_query)
    _emit_progress(state, progress_callback, 5, "读取用户输入政策", "正在读取政策链接或正文")
    run_policy_analyst(state, store, mcp_invoker=mcp_invoker, progress_callback=progress_callback)
    _emit_progress(state, progress_callback, 30, "政策分析", "已完成政策身份、措施和力度分析")
    run_impact_analyst(state, mcp_invoker=mcp_invoker)
    _emit_progress(state, progress_callback, 55, "行业影响分析", "已完成实施路径和行业影响分析")
    run_company_matcher(state, mcp_invoker=mcp_invoker, use_annual_reports=use_annual_reports)
    _emit_progress(state, progress_callback, 75, "公司业务匹配", "已完成候选公司业务匹配")
    _emit_progress(state, progress_callback, 90, "生成报告", "正在整合报告")
    write_research_report(state)
    _emit_progress(state, progress_callback, 100, "完成", "报告已生成")
    return state


def run_llm_policy_research_workflow(
    user_query: str,
    store: SQLitePolicyStore,
    llm_client: LLMClient | None = None,
    mcp_invoker: MCPToolInvoker | None = None,
    use_annual_reports: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> PolicyResearchState:
    """Run the optional LLM-backed workflow over retrieved evidence."""

    state = PolicyResearchState(user_query=user_query)
    client = llm_client or create_llm_client()
    _emit_progress(state, progress_callback, 5, "读取用户输入政策", "正在读取政策链接或正文")
    run_llm_policy_analyst(
        state,
        store,
        llm_client=client,
        mcp_invoker=mcp_invoker,
        progress_callback=progress_callback,
    )
    _emit_progress(state, progress_callback, 30, "政策分析", "已完成 Policy Analyst 分析")
    run_llm_impact_analyst(state, llm_client=client, mcp_invoker=mcp_invoker)
    _emit_progress(state, progress_callback, 55, "行业影响分析", "已完成 Impact Analyst 分析")
    run_llm_company_matcher(
        state,
        llm_client=client,
        mcp_invoker=mcp_invoker,
        use_annual_reports=use_annual_reports,
    )
    _emit_progress(state, progress_callback, 75, "公司业务匹配", "已完成 Company Matcher 分析")
    _emit_progress(state, progress_callback, 90, "生成报告", "正在整合报告")
    write_research_report(state)
    _emit_progress(state, progress_callback, 100, "完成", "报告已生成")
    return state


def _emit_progress(
    state: PolicyResearchState,
    callback: ProgressCallback | None,
    progress: int,
    stage: str,
    message: str,
) -> None:
    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "progress": progress,
        "stage": stage,
        "message": message,
    }
    state.progress_events.append(event)
    if callback:
        callback(progress, stage, message)
