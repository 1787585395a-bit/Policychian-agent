from __future__ import annotations

import json
import re
from typing import Any

from policychain.llm import LLMClient, observed_llm_generate
from policychain.observability import current_run_recorder, record_event
from policychain.prompts import render_prompt
from policychain.safety import assert_no_investment_advice
from policychain.state import PolicyResearchState


class ReportWriterError(RuntimeError):
    """Raised when a final report cannot be generated safely."""


REFERENCE_KEY_EVIDENCE_LIMIT = 2
REFERENCE_EXTERNAL_EVIDENCE_LIMIT = 2
REFERENCE_TOOL_LOG_LIMIT = 4


def write_research_report(state: PolicyResearchState) -> str:
    """Write a detailed deterministic fallback report."""

    report = "\n\n".join(
        [
            "# PolicyChain 政策研究报告",
            _summary_section(state),
            _policy_reading_section(state.policy_analysis),
            _similar_policy_section(state.similar_policy_matches),
            _impact_path_section(state.implementation_path, state.industry_impacts),
            _company_matches_section(state.industry_impacts, state.company_matches, state.company_coverage),
            _uncertainty_section(state.uncertainties),
            _reference_appendix(state),
            "本报告仅用于政策研究和公司业务匹配分析，不构成任何投资建议。",
        ]
    ).strip()
    assert_no_investment_advice(report, context="Report")
    state.final_report = report
    record_event("report.source", stage="report_writer", status="ok", source="deterministic_rules")
    return report


def write_llm_research_report(state: PolicyResearchState, llm_client: LLMClient) -> str:
    """Ask an LLM to write a natural Markdown report, then append compact references."""

    try:
        coverage_matrix = _coverage_matrix(state)
        prompt = render_prompt(
            "report_writer",
            policy_analysis=_json_for_prompt(state.policy_analysis),
            impact_analysis=_json_for_prompt(
                {
                    "implementation_path": state.implementation_path,
                    "industry_impacts": state.industry_impacts,
                    "coverage_matrix": coverage_matrix,
                    "industry_research_summary": _compact_external_evidence(state.industry_research, limit=4),
                    "react_trace_summary": _compact_react_traces(state.react_traces, limit=4),
                    "tool_call_summary": _compact_tool_logs(state.tool_call_logs, limit=4),
                }
            ),
            company_matches=_json_for_prompt(
                {
                    "company_candidates": state.company_candidates,
                    "company_matches": state.company_matches,
                    "company_coverage": coverage_matrix,
                    "company_audit": state.company_match_audit[:12],
                    "company_research_summary": _compact_external_evidence(state.company_research, limit=4),
                }
            ),
            evidence=_json_for_prompt(_reference_payload(state)),
            uncertainties=_json_for_prompt(state.uncertainties),
        )
        body = _strip_reference_sections(
            observed_llm_generate(llm_client, prompt["system"], prompt["user"], agent="report_writer")
        )
        if not body.strip():
            raise ReportWriterError("LLM Report Writer returned an empty report")
        body = _ensure_impact_coverage(body.strip(), state)
        report = "\n\n".join(
            [
                body,
                _reference_appendix(state),
                "本报告仅用于政策研究和公司业务匹配分析，不构成任何投资建议。",
            ]
        ).strip()
        assert_no_investment_advice(report, context="LLM report")
        state.final_report = report
        record_event("report.source", stage="report_writer", status="ok", source="llm")
        return report
    except Exception as exc:
        recorder = current_run_recorder()
        if recorder is not None:
            recorder.mark_fallback("report_writer", str(exc)[:300], "deterministic_report_writer")
        state.uncertainties = _unique(
            [
                *state.uncertainties,
                f"LLM Report Writer 未能完成自然语言报告生成，已回退到确定性报告：{exc}",
            ]
        )
        return write_research_report(state)


def _summary_section(state: PolicyResearchState) -> str:
    identity = state.policy_analysis.get("policy_identity") or {}
    title = identity.get("title") or "用户输入政策"
    agencies = ", ".join(identity.get("issuing_agencies") or []) or "发布主体待核验"
    strength = (state.policy_analysis.get("strength_assessment") or {}).get("level") or "unknown"
    impact_count = len(state.industry_impacts)
    matched_path_count = sum(1 for item in _coverage_matrix(state) if item.get("passed_count"))
    return (
        "## 研究摘要\n\n"
        f"本次分析以用户输入政策为主对象。当前识别的政策是“{title}”，发布主体为 {agencies}，"
        f"政策力度初步判断为 {strength}。工作流形成了 {impact_count} 条行业影响路径，其中 "
        f"{matched_path_count} 条路径形成了通过业务审查的 A 股公司匹配，其余路径会说明暂未形成可靠匹配的原因。"
    )


def _policy_reading_section(policy_analysis: dict[str, Any]) -> str:
    identity = policy_analysis.get("policy_identity") or {}
    if identity.get("status") == "no_policy_found":
        return "## 主政策解读\n\n未检索到可用于政策分析的主政策正文，无法形成可靠解读。"
    strength = policy_analysis.get("strength_assessment") or {}
    goals = _join_items(policy_analysis.get("policy_goals") or [], "政策目标待提取")
    measures = _join_items(policy_analysis.get("policy_measures") or [], "主要措施待提取")
    targets = _join_items(policy_analysis.get("target_entities") or [], "约束或影响对象待提取")
    reasons = _join_items(strength.get("reasons") or [], "力度判断依据待补充")
    return (
        "## 主政策解读\n\n"
        f"政策标题：{identity.get('title') or '未知'}。\n"
        f"文号：{identity.get('document_number') or '未知'}；发布日期：{identity.get('publish_date') or '未知'}；"
        f"来源：{identity.get('source_url') or '用户粘贴文本或本地样例'}。\n\n"
        f"从政策内容看，政策目标主要是：{goals}。它直接约束或影响的对象包括：{targets}。"
        f"核心措施可以概括为：{measures}。力度判断为 {strength.get('level') or 'unknown'}，主要依据是：{reasons}。"
        "后续行业和公司分析均以这些措施如何落地为起点，而不是直接从概念板块名称推出结论。"
    )


def _similar_policy_section(similar_policy_matches: list[dict[str, Any]]) -> str:
    if not similar_policy_matches:
        return (
            "## 相似政策对比\n\n"
            "未在本地知识库中找到相似政策。若启用 Web Search，官方解读、上位政策或后续配套材料会在末尾参考资料中简要列出；"
            "它们只能补充对比，不能替代用户输入政策。"
        )

    lines = [
        "## 相似政策对比",
        "",
        "本地知识库中的相似政策只作为对照。对比重点放在发布层级、政策工具、执行力度、适用对象和时间演进，而不是简单判断“相同/不同”。",
    ]
    for item in similar_policy_matches[:5]:
        title = str(item.get("title") or item.get("policy_id") or "未命名政策")
        agency = str(item.get("agency") or item.get("issuing_agency") or "")
        text = str(item.get("matched_text") or item.get("section_title") or "")
        source = str(item.get("source_url") or item.get("url") or item.get("policy_id") or "")
        lines.extend(
            [
                "",
                f"- {title}",
                f"  - 层级/主体：{_infer_level_from_text(' '.join([title, agency]))}；{agency or '知识库未记录发布主体'}",
                f"  - 政策工具：{_infer_policy_tool(' '.join([title, text]))}",
                f"  - 力度特征：{_infer_strength(' '.join([title, text]))}",
            ]
        )
        if item.get("publish_date"):
            lines.append(f"  - 发布时间：{item.get('publish_date')}")
        if source:
            lines.append(f"  - 来源：{source}")
        if text:
            lines.append(f"  - 可比内容：{_clip(text, 180)}")
    return "\n".join(lines)


def _impact_path_section(implementation_path: list[dict[str, Any]], industry_impacts: list[dict[str, Any]]) -> str:
    lines = [
        "## 实施路径与行业影响",
        "",
        "行业判断从政策措施出发，经由实施主体和实施行为，再落到产业链环节与经营变量。每条路径都需要说明影响方向、时间差异、成立条件和风险。",
    ]
    if not implementation_path and not industry_impacts:
        lines.append("尚未从政策文本中形成清晰实施路径和行业影响。")
        return "\n".join(lines)

    if implementation_path:
        lines.append("")
        lines.append("政策落地链条可以概括为：")
        for step in implementation_path[:6]:
            lines.append(
                f"- 路径步骤 {step.get('step_index')}: {step.get('actor') or '相关主体'}执行"
                f"“{step.get('action') or '政策要求'}”，通过{step.get('mechanism') or '政策传导机制'}影响产业活动。"
            )

    for index, impact in enumerate(industry_impacts, start=1):
        variables = _join_items(impact.get("business_variables") or [], "经营变量待核验")
        company_types = _join_items(impact.get("affected_company_types") or [], "公司类型待核验")
        conditions = _join_items(impact.get("conditions") or [], "需要后续配套政策、执行口径和产业数据验证")
        risks = _join_items(impact.get("risks") or [], "主要风险待补充")
        lines.extend(
            [
                "",
                f"### 路径 {index}: {impact.get('industry') or '相关行业'}",
                (
                    f"政策措施是“{impact.get('policy_measure') or '待细化'}”，落地后对应的实施行为是"
                    f"“{impact.get('implementation_action') or '待细化'}”。这条路径影响的产业链环节是"
                    f"“{impact.get('chain_segment') or '待识别'}”，影响类型为 {impact.get('impact_type') or 'unknown'}，"
                    f"方向为 {impact.get('direction') or 'unknown'}。"
                ),
                (
                    f"传导逻辑是：{impact.get('transmission_logic') or '暂无明确传导解释'}。"
                    f"具体到经营变量，重点观察 {variables}；可能受影响的公司类型包括 {company_types}。"
                ),
                (
                    f"时间维度上，短期通常体现为合规、技改、项目申报、资本开支或订单确认节奏的变化；"
                    f"中长期则取决于政策是否形成财政资金、示范项目、准入规则、标准体系或需求扩散。"
                    f"这一路径成立的条件是：{conditions}。主要风险是：{risks}。"
                ),
            ]
        )
    return "\n".join(lines)


def _company_matches_section(
    industry_impacts: list[dict[str, Any]],
    company_matches: list[dict[str, Any]],
    company_coverage: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        "## A 股公司业务匹配",
        "",
        "以下内容是业务匹配或关注清单，不是投资建议。匹配依据是公司主营业务、产品服务、公告/官网描述或 CNFinancial 返回的公开业务资料是否能对应前述行业路径。"
        "仅有概念板块标签、缺少业务证据的公司会被降为低置信或剔除。",
    ]
    if not industry_impacts:
        lines.append("尚未形成行业影响路径，因此无法进行逐路径公司匹配。")
        return "\n".join(lines)

    coverage = company_coverage or _fallback_coverage(industry_impacts, company_matches)
    matches_by_impact = _matches_by_impact(company_matches)
    for index, impact in enumerate(industry_impacts, start=1):
        impact_id = f"IMP-{index:03d}"
        coverage_item = next((item for item in coverage if item.get("impact_id") == impact_id), {})
        scoped_matches = matches_by_impact.get(impact_id, [])[:3]
        lines.extend(
            [
                "",
                f"### {impact_id}：{impact.get('industry') or coverage_item.get('industry') or '相关行业'}",
                (
                    f"对应产业链环节为“{impact.get('chain_segment') or coverage_item.get('chain_segment') or '待识别'}”，"
                    f"关键经营变量包括：{_join_items(impact.get('business_variables') or coverage_item.get('business_variables') or [], '待核验')}。"
                ),
            ]
        )
        if not scoped_matches:
            reason = coverage_item.get("no_match_reason") or "暂未形成可靠 A 股公司匹配，可能是候选不足、业务证据不足或路径过宽。"
            lines.append(f"暂未形成可靠 A 股公司匹配。原因：{reason}")
            continue

        for company in scoped_matches:
            evidence = company.get("business_evidence") or []
            evidence_text = _clip(str(evidence[0].get("text") if evidence else ""), 180) if evidence else "暂无业务证据摘要"
            risk_text = _join_items(company.get("negative_evidence") or company.get("risks") or [], "暂无明确反面证据")
            audit_reason = company.get("audit_reason") or "已通过业务相关性审查"
            lines.extend(
                [
                    "",
                    f"- {company.get('company_name') or '未知公司'}（{company.get('stock_code') or '代码未知'}）",
                    f"  - 匹配环节：{company.get('chain_segment') or company.get('industry_segment') or '待识别'}",
                    f"  - 相关业务：{company.get('matched_business') or company.get('related_product_or_business') or '待验证'}",
                    (
                        f"  - 传导机制：该业务与本路径的经营变量存在交集，因此政策若通过"
                        f"“{impact.get('implementation_action') or '实施行为'}”落地，可能影响其需求、成本、合规投入、订单或项目机会。"
                    ),
                    f"  - 审查结论：{audit_reason}",
                    f"  - 匹配等级/置信度：{company.get('match_level') or 'low'} / {company.get('confidence')}",
                    f"  - 业务证据：{evidence_text}",
                    f"  - 反面证据或风险：{risk_text}",
                ]
            )
    return "\n".join(lines)


def _uncertainty_section(uncertainties: list[str]) -> str:
    lines = ["## 不确定性与使用边界"]
    if not uncertainties:
        lines.append("当前未记录额外不确定性，但仍需要结合后续配套政策、行业数据和公司原始公告持续验证。")
    else:
        lines.extend(f"- {item}" for item in uncertainties[:10])
    return "\n".join(lines)


def _reference_appendix(state: PolicyResearchState) -> str:
    lines = ["## 参考资料与工具依据"]

    key_evidence = state.evidence[:REFERENCE_KEY_EVIDENCE_LIMIT]
    if key_evidence:
        lines.append("")
        lines.append("本地政策证据：")
        for item in key_evidence:
            source = item.get("source_url") or item.get("policy_id") or "本地政策文本"
            lines.append(f"- {source} / {item.get('chunk_id') or 'chunk'}：{_clip(str(item.get('text') or ''), 100)}")

    external_evidence = _prioritize_external_evidence(state.external_evidence)[:REFERENCE_EXTERNAL_EVIDENCE_LIMIT]
    if external_evidence:
        lines.append("")
        lines.append("外部资料：")
        for item in external_evidence:
            source = item.get("server_name") or item.get("source_name") or "external"
            tool = item.get("tool_name") or "source"
            title = item.get("title") or item.get("company_name") or item.get("source_org") or "未命名资料"
            url = item.get("source_url") or ""
            date = item.get("published_date") or item.get("data_date") or ""
            suffix = f"（{date}）" if date else ""
            link = f" {url}" if url else ""
            lines.append(f"- {source}.{tool}：{title}{suffix}{link}")

    tool_summaries = _compact_tool_logs(state.tool_call_logs, limit=REFERENCE_TOOL_LOG_LIMIT)
    if not tool_summaries and state.react_traces:
        tool_summaries = _compact_react_traces(state.react_traces, limit=REFERENCE_TOOL_LOG_LIMIT)
    if tool_summaries:
        lines.append("")
        lines.append("工具调用摘要：")
        lines.extend(f"- {item}" for item in tool_summaries)

    if len(lines) == 1:
        lines.append("暂无可列出的参考资料或工具依据。")
    return "\n".join(lines)


def _coverage_matrix(state: PolicyResearchState) -> list[dict[str, Any]]:
    if state.company_coverage:
        return state.company_coverage
    return _fallback_coverage(state.industry_impacts, state.company_matches)


def _fallback_coverage(industry_impacts: list[dict[str, Any]], company_matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches_by_impact = _matches_by_impact(company_matches)
    coverage: list[dict[str, Any]] = []
    for index, impact in enumerate(industry_impacts, start=1):
        impact_id = f"IMP-{index:03d}"
        scoped = matches_by_impact.get(impact_id, [])
        coverage.append(
            {
                "impact_id": impact_id,
                "industry": impact.get("industry") or "",
                "chain_segment": impact.get("chain_segment") or "",
                "business_variables": list(impact.get("business_variables") or []),
                "affected_company_types": list(impact.get("affected_company_types") or []),
                "candidate_count": len(company_matches),
                "passed_count": len(scoped),
                "rejected_count": 0,
                "company_names": [item.get("company_name") for item in scoped if item.get("company_name")],
                "no_match_reason": "" if scoped else "暂未形成可靠 A 股公司匹配，缺少通过路径绑定审查的公司。",
            }
        )
    return coverage


def _matches_by_impact(company_matches: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, company in enumerate(company_matches, start=1):
        impact_id = str(company.get("impact_id") or "")
        if not impact_id:
            impact_id = "IMP-001" if len(company_matches) == 1 else f"IMP-{index:03d}"
        grouped.setdefault(impact_id, []).append(company)
    for items in grouped.values():
        items.sort(key=lambda item: float(item.get("confidence") or 0), reverse=True)
    return grouped


def _ensure_impact_coverage(body: str, state: PolicyResearchState) -> str:
    missing: list[dict[str, Any]] = []
    for impact in state.industry_impacts:
        industry = str(impact.get("industry") or "")
        chain = str(impact.get("chain_segment") or "")
        if industry and industry in body:
            continue
        if chain and chain in body:
            continue
        missing.append(impact)
    if not missing:
        return body

    supplement_state = PolicyResearchState(user_query=state.user_query)
    supplement_state.industry_impacts = missing
    supplement_state.company_matches = state.company_matches
    supplement_state.company_coverage = [
        item
        for item in _coverage_matrix(state)
        if item.get("industry") in {impact.get("industry") for impact in missing}
    ]
    supplement = "\n\n".join(
        [
            "## 补充：未充分展开的行业路径",
            _impact_path_section([], missing).replace("## 实施路径与行业影响\n\n", ""),
            _company_matches_section(missing, state.company_matches, supplement_state.company_coverage),
        ]
    )
    return f"{body}\n\n{supplement}"


def _reference_payload(state: PolicyResearchState) -> dict[str, Any]:
    return {
        "key_evidence": state.evidence[:REFERENCE_KEY_EVIDENCE_LIMIT],
        "external_evidence": _compact_external_evidence(state.external_evidence, limit=REFERENCE_EXTERNAL_EVIDENCE_LIMIT),
        "tool_calls": _compact_tool_logs(state.tool_call_logs, limit=REFERENCE_TOOL_LOG_LIMIT),
        "react_traces": _compact_react_traces(state.react_traces, limit=REFERENCE_TOOL_LOG_LIMIT),
    }


def _compact_external_evidence(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    output = []
    for item in _prioritize_external_evidence(items)[:limit]:
        output.append(
            {
                "server_name": item.get("server_name"),
                "tool_name": item.get("tool_name"),
                "title": item.get("title") or item.get("company_name"),
                "source_org": item.get("source_org"),
                "published_date": item.get("published_date") or item.get("data_date"),
                "source_url": item.get("source_url"),
                "summary": _clip(str(item.get("summary") or item.get("text") or ""), 120),
            }
        )
    return output


def _compact_tool_logs(tool_logs: list[dict[str, Any]], limit: int) -> list[str]:
    output = []
    for log in _prioritize_tool_logs(tool_logs)[:limit]:
        server = log.get("server_name") or "server"
        tool = log.get("tool_name") or "tool"
        status = log.get("status") or "unknown"
        count = log.get("count")
        error = log.get("error")
        args = _compact_json(log.get("arguments") or {})
        summary = f"{server}.{tool} {args} -> {status}, count={count}"
        if error:
            summary += f", error={_clip(str(error), 100)}"
        output.append(summary)
    return output


def _compact_react_traces(react_traces: list[dict[str, Any]], limit: int) -> list[str]:
    output = []
    for trace in react_traces:
        if _is_catalog_tool(str(trace.get("action") or "")):
            continue
        stage = trace.get("stage") or "react"
        action = trace.get("action") or "unknown"
        count = trace.get("evidence_count")
        detail = trace.get("error") or trace.get("observation") or trace.get("message") or ""
        output.append(f"{stage}.{action} -> count={count}; {_clip(str(detail), 100)}")
        if len(output) >= limit:
            break
    return output


def _prioritize_external_evidence(external_evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "cn-financial": [],
        "web-search": [],
        "other": [],
    }
    for item in external_evidence:
        if _is_catalog_tool(str(item.get("tool_name") or "")):
            continue
        server = str(item.get("server_name") or item.get("source_name") or "").lower()
        grouped[server if server in grouped else "other"].append(item)

    output: list[dict[str, Any]] = []
    for server_name in ("cn-financial", "web-search", "other"):
        output.extend(grouped[server_name])
    return output


def _prioritize_tool_logs(tool_logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"cn-financial": 0, "policychain": 1, "web-search": 2}
    visible = [
        item
        for item in tool_logs
        if not item.get("internal_only") and not _is_catalog_tool(str(item.get("tool_name") or ""))
    ]
    return sorted(visible, key=lambda item: order.get(str(item.get("server_name") or ""), 9))


def _is_catalog_tool(tool_name: str) -> bool:
    normalized = tool_name.rsplit(".", 1)[-1]
    return normalized in {"get_industry_list", "get_concept_list"}


def _strip_reference_sections(text: str) -> str:
    cleaned = text.strip()
    for marker in ("## 参考资料", "## 关键证据", "## 外部证据", "## ReAct", "## MCP"):
        index = cleaned.find(marker)
        if index > 0:
            cleaned = cleaned[:index].rstrip()
    return cleaned


def _infer_level_from_text(text: str) -> str:
    if any(term in text for term in ("国务院", "国家", "部", "委", "网信办")):
        return "国家级或部委层面"
    if any(term in text for term in ("省", "自治区", "直辖市")):
        return "省级地方政策"
    if any(term in text for term in ("市", "区", "县")):
        return "市县级地方政策"
    return "层级待核验"


def _infer_policy_tool(text: str) -> str:
    tools = []
    if any(term in text for term in ("补贴", "资金", "奖励", "贴息")):
        tools.append("资金支持")
    if any(term in text for term in ("试点", "示范", "名单")):
        tools.append("试点示范")
    if any(term in text for term in ("备案", "许可", "审批", "评估", "监管")):
        tools.append("准入或监管")
    if any(term in text for term in ("鼓励", "支持", "促进")):
        tools.append("鼓励引导")
    return "、".join(tools) if tools else "政策工具待核验"


def _infer_strength(text: str) -> str:
    if any(term in text for term in ("应当", "必须", "不得", "禁止", "处罚", "整改")):
        return "约束性较强"
    if any(term in text for term in ("试点", "示范", "申报", "补贴", "资金")):
        return "执行抓手较明确"
    if any(term in text for term in ("鼓励", "支持", "引导")):
        return "引导性较强"
    return "力度待核验"


def _join_items(values: list[Any], fallback: str) -> str:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    return "；".join(cleaned[:8]) if cleaned else fallback


def _clip(text: str, max_chars: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip() + "..."


def _json_for_prompt(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _compact_json(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return _clip(text, 100)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        compacted = re.sub(r"\s+", " ", str(value)).strip()
        if compacted and compacted not in seen:
            seen.add(compacted)
            output.append(compacted)
    return output
