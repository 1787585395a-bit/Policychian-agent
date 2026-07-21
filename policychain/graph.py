from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import re

from policychain.agents import (
    run_company_matcher,
    run_impact_analyst,
    run_llm_company_matcher,
    run_llm_impact_analyst,
    run_llm_policy_analyst,
    run_policy_analyst,
)
from policychain.agents.report_writer import write_llm_research_report, write_research_report
from policychain.llm import LLMClient, LLMConfigurationError, create_llm_client
from policychain.mcp import MCPToolInvoker
from policychain.observability import RunRecorder, record_event
from policychain.state import PolicyResearchState
from policychain.storage.sqlite_store import SQLitePolicyStore


ProgressCallback = Callable[[int, str, str], None]


def run_policy_research_workflow(
    user_query: str,
    store: SQLitePolicyStore,
    mcp_invoker: MCPToolInvoker | None = None,
    progress_callback: ProgressCallback | None = None,
    run_recorder: RunRecorder | None = None,
) -> PolicyResearchState:
    recorder = run_recorder or RunRecorder(mode="deterministic")
    state = _initialize_run_state(user_query, recorder)
    with recorder.activate():
        try:
            _run_deterministic_stages(state, store, recorder, mcp_invoker, progress_callback)
        except Exception as exc:
            _sync_run_state(state, recorder)
            recorder.finish("failed", error=_compact_error(exc))
            raise
        _sync_run_state(state, recorder)
        recorder.finish("completed", report_source="deterministic_rules")
    return state


def run_llm_policy_research_workflow(
    user_query: str,
    store: SQLitePolicyStore,
    llm_client: LLMClient | None = None,
    mcp_invoker: MCPToolInvoker | None = None,
    progress_callback: ProgressCallback | None = None,
    run_recorder: RunRecorder | None = None,
) -> PolicyResearchState:
    """Run the optional LLM-backed workflow over retrieved evidence."""

    recorder = run_recorder or RunRecorder(mode="llm")
    state = _initialize_run_state(user_query, recorder)
    with recorder.activate():
        try:
            try:
                client = llm_client or create_llm_client()
            except LLMConfigurationError as exc:
                recorder.mark_fallback("workflow", _compact_error(exc), "deterministic")
                recorder.mode = "deterministic"
                state.run_mode = "deterministic"
                _append_uncertainty(state, f"LLM 初始化失败，已回退到确定性流程：{_compact_error(exc)}")
                _emit_progress(state, progress_callback, 3, "模型初始化", "LLM 初始化失败，正在使用确定性流程")
                _run_deterministic_stages(state, store, recorder, mcp_invoker, progress_callback)
                _sync_run_state(state, recorder)
                recorder.finish("completed", report_source="deterministic_rules")
                return state
            _emit_progress(state, progress_callback, 5, "读取用户输入政策", "正在读取政策链接或正文")

            try:
                _run_recorded_agent(
                    recorder,
                    "policy_analyst",
                    lambda: run_llm_policy_analyst(
                        state,
                        store,
                        llm_client=client,
                        mcp_invoker=mcp_invoker,
                        progress_callback=progress_callback,
                    ),
                )
            except Exception as exc:
                recorder.mark_fallback("policy_analyst", _compact_error(exc), "deterministic_policy_analyst")
                _append_uncertainty(state, f"LLM Policy Analyst 失败，已回退到确定性政策分析：{_compact_error(exc)}")
                _emit_progress(state, progress_callback, 24, "政策分析回退", "LLM 政策分析失败，正在使用确定性流程")
                _run_recorded_agent(
                    recorder,
                    "policy_analyst_fallback",
                    lambda: run_policy_analyst(state, store, mcp_invoker=mcp_invoker, progress_callback=progress_callback),
                )
            _emit_progress(state, progress_callback, 30, "政策分析", "已完成 Policy Analyst 分析")

            try:
                _run_recorded_agent(
                    recorder,
                    "impact_analyst",
                    lambda: run_llm_impact_analyst(state, llm_client=client, mcp_invoker=mcp_invoker),
                )
            except Exception as exc:
                recorder.mark_fallback("impact_analyst", _compact_error(exc), "deterministic_impact_analyst")
                _append_uncertainty(state, f"LLM Impact Analyst 失败，已回退到确定性行业影响分析：{_compact_error(exc)}")
                _emit_progress(state, progress_callback, 50, "行业影响分析回退", "LLM 行业影响分析失败，正在使用确定性流程")
                _run_recorded_agent(recorder, "impact_analyst_fallback", lambda: run_impact_analyst(state, mcp_invoker=mcp_invoker))
            _emit_progress(state, progress_callback, 55, "行业影响分析", "已完成 Impact Analyst 分析")

            try:
                _run_recorded_agent(
                    recorder,
                    "company_matcher",
                    lambda: run_llm_company_matcher(state, llm_client=client, mcp_invoker=mcp_invoker),
                )
            except Exception as exc:
                recorder.mark_fallback("company_matcher", _compact_error(exc), "deterministic_company_matcher")
                _append_uncertainty(state, f"LLM Company Matcher 失败，已回退到确定性公司业务匹配：{_compact_error(exc)}")
                _emit_progress(state, progress_callback, 70, "公司业务匹配回退", "LLM 公司匹配失败，正在使用确定性流程")
                _run_recorded_agent(recorder, "company_matcher_fallback", lambda: run_company_matcher(state, mcp_invoker=mcp_invoker))
            _emit_company_coverage_progress(state, progress_callback)
            _emit_progress(state, progress_callback, 75, "公司业务匹配", "已完成 Company Matcher 分析")

            _emit_progress(state, progress_callback, 90, "生成报告", "正在整合报告")
            if _can_use_llm_report(state):
                _run_recorded_agent(recorder, "report_writer", lambda: write_llm_research_report(state, client))
            else:
                _run_recorded_agent(recorder, "report_writer", lambda: write_research_report(state))
                record_event("report.source", stage="report_writer", status="ok", source="deterministic_rules")
            _emit_progress(state, progress_callback, 100, "完成", "报告已生成")
        except Exception as exc:
            _sync_run_state(state, recorder)
            recorder.finish("failed", error=_compact_error(exc))
            raise
        _sync_run_state(state, recorder)
        recorder.finish("completed")
    return state


def _can_use_llm_report(state: PolicyResearchState) -> bool:
    return bool(state.policy_analysis) and state.policy_analysis.get("policy_identity", {}).get("status") != "no_policy_found"


def _run_deterministic_stages(
    state: PolicyResearchState,
    store: SQLitePolicyStore,
    recorder: RunRecorder,
    mcp_invoker: MCPToolInvoker | None,
    progress_callback: ProgressCallback | None,
) -> None:
    _emit_progress(state, progress_callback, 5, "读取用户输入政策", "正在读取政策链接或正文")
    _run_recorded_agent(
        recorder,
        "policy_analyst",
        lambda: run_policy_analyst(state, store, mcp_invoker=mcp_invoker, progress_callback=progress_callback),
    )
    _emit_progress(state, progress_callback, 30, "政策分析", "已完成政策身份、措施和力度分析")
    _run_recorded_agent(recorder, "impact_analyst", lambda: run_impact_analyst(state, mcp_invoker=mcp_invoker))
    _emit_progress(state, progress_callback, 55, "行业影响分析", "已完成实施路径和行业影响分析")
    _run_recorded_agent(recorder, "company_matcher", lambda: run_company_matcher(state, mcp_invoker=mcp_invoker))
    _emit_company_coverage_progress(state, progress_callback)
    _emit_progress(state, progress_callback, 75, "公司业务匹配", "已完成候选公司业务匹配")
    _emit_progress(state, progress_callback, 90, "生成报告", "正在整合报告")
    _run_recorded_agent(recorder, "report_writer", lambda: write_research_report(state))
    record_event("report.source", stage="report_writer", status="ok", source="deterministic_rules")
    _emit_progress(state, progress_callback, 100, "完成", "报告已生成")


def _initialize_run_state(user_query: str, recorder: RunRecorder) -> PolicyResearchState:
    state = PolicyResearchState(user_query=user_query)
    state.run_id = recorder.run_id
    state.run_mode = recorder.mode
    return state


def _run_recorded_agent(recorder: RunRecorder, agent: str, call: Callable[[], object]) -> object:
    recorder.set_agent_status(agent, "running")
    try:
        result = call()
    except Exception as exc:
        recorder.set_agent_status(agent, "failed", error=_compact_error(exc))
        raise
    recorder.set_agent_status(agent, "completed")
    return result


def _sync_run_state(state: PolicyResearchState, recorder: RunRecorder) -> None:
    state.run_id = recorder.run_id
    state.run_mode = recorder.mode
    state.agent_status = dict(recorder.agent_status)
    state.fallback_used = recorder.fallback_used


def _emit_company_coverage_progress(
    state: PolicyResearchState,
    callback: ProgressCallback | None,
) -> None:
    if not state.company_coverage:
        return
    path_count = len(state.company_coverage)
    passed = sum(int(item.get("passed_count") or 0) for item in state.company_coverage)
    rejected = sum(int(item.get("rejected_count") or 0) for item in state.company_coverage)
    no_match = sum(1 for item in state.company_coverage if not item.get("passed_count"))
    _emit_progress(
        state,
        callback,
        72,
        "公司匹配审查",
        f"已审查 {path_count} 条行业路径，通过 {passed} 个公司匹配，剔除 {rejected} 个候选，{no_match} 条路径暂未形成可靠公司匹配",
    )


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
    record_event("workflow.progress", stage=stage, status="ok", progress=progress, message=message)
    if callback:
        callback(progress, stage, message)


def _append_uncertainty(state: PolicyResearchState, value: str) -> None:
    state.uncertainties = _unique([*state.uncertainties, value])


def _compact_error(exc: Exception) -> str:
    return re.sub(r"\s+", " ", str(exc)).strip()[:300] or exc.__class__.__name__


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value)).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output
