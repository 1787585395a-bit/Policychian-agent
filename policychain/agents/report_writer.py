from __future__ import annotations

from typing import Any

from policychain.safety import assert_no_investment_advice
from policychain.state import PolicyResearchState


class ReportWriterError(RuntimeError):
    """Raised when a final report cannot be generated safely."""


def write_research_report(state: PolicyResearchState) -> str:
    report = "\n\n".join(
        [
            "# PolicyChain 政策研究报告",
            _policy_identity_section(state.policy_analysis),
            _similar_policy_section(state.similar_policy_matches),
            _policy_analysis_section(state.policy_analysis),
            _implementation_section(state.implementation_path),
            _industry_impacts_section(state.industry_impacts),
            _company_matches_section(state.company_matches),
            _evidence_section(state.evidence),
            _external_evidence_section(state.external_evidence),
            _uncertainty_section(state.uncertainties),
            "本报告仅用于政策研究和业务匹配分析，不构成任何投资建议。",
        ]
    ).strip()
    assert_no_investment_advice(report, context="Report")
    state.final_report = report
    return report


def _policy_identity_section(policy_analysis: dict[str, Any]) -> str:
    identity = policy_analysis.get("policy_identity") or {}
    if not identity:
        return "## 1. 政策基本信息\n\n未形成政策基本信息。"
    lines = [
        "## 1. 政策基本信息",
        f"- 标题：{identity.get('title') or '未知'}",
        f"- Policy ID：{identity.get('policy_id') or '未知'}",
        f"- 文号：{identity.get('document_number') or '未知'}",
        f"- 发布日期：{identity.get('publish_date') or '未知'}",
        f"- 发布机构：{', '.join(identity.get('issuing_agencies') or []) or '未知'}",
        f"- 来源：{identity.get('source_url') or '未知'}",
    ]
    return "\n".join(lines)


def _policy_analysis_section(policy_analysis: dict[str, Any]) -> str:
    strength = policy_analysis.get("strength_assessment") or {}
    lines = [
        "## 2. 政策核心内容与力度判断",
        "### 政策目标",
        *_bullets(policy_analysis.get("policy_goals") or []),
        "### 主要措施",
        *_bullets(policy_analysis.get("policy_measures") or []),
        "### 力度判断",
        f"- 等级：{strength.get('level') or 'unknown'}",
        *_bullets(strength.get("reasons") or []),
    ]
    return "\n".join(lines)


def _similar_policy_section(similar_policy_matches: list[dict[str, Any]]) -> str:
    lines = ["## \u76f8\u4f3c\u653f\u7b56\u5bf9\u6bd4"]
    if not similar_policy_matches:
        return "\n".join([*lines, "- \u672a\u5728\u672c\u5730\u77e5\u8bc6\u5e93\u4e2d\u627e\u5230\u76f8\u4f3c\u653f\u7b56\u3002"])
    for item in similar_policy_matches[:6]:
        title = item.get("title") or item.get("policy_id") or "unknown"
        date = item.get("publish_date") or ""
        agency = item.get("agency") or ""
        score = item.get("score")
        text = item.get("matched_text") or ""
        detail = f"- {title}"
        if date:
            detail += f" / {date}"
        if agency:
            detail += f" / {agency}"
        if score is not None:
            detail += f" / score={score}"
        if text:
            detail += f"\uff1a{_clip(str(text), 160)}"
        lines.append(detail)
    return "\n".join(lines)


def _implementation_section(implementation_path: list[dict[str, Any]]) -> str:
    lines = ["## 3. 实施路径分析"]
    if not implementation_path:
        return "\n".join([*lines, "- 尚未形成实施路径。"])
    for step in implementation_path:
        lines.append(
            f"- Step {step.get('step_index')}: {step.get('actor')} 通过「{step.get('mechanism')}」落实：{step.get('action')}"
        )
    return "\n".join(lines)


def _industry_impacts_section(industry_impacts: list[dict[str, Any]]) -> str:
    lines = ["## 4. 行业影响分析"]
    if not industry_impacts:
        return "\n".join([*lines, "- 尚未形成行业影响分析。"])
    for impact in industry_impacts:
        lines.append(
            f"- {impact.get('industry')}（{impact.get('impact_type')}/{impact.get('direction')}）：{impact.get('transmission_logic')}"
        )
    return "\n".join(lines)


def _company_matches_section(company_matches: list[dict[str, Any]]) -> str:
    lines = ["## 5. 公司业务匹配清单"]
    if not company_matches:
        return "\n".join([*lines, "- 尚未形成公司业务匹配。"])
    for company in company_matches:
        lines.append(
            f"- {company.get('company_name')}：{company.get('matched_business')}；匹配等级：{company.get('match_level')}；置信度：{company.get('confidence')}"
        )
    return "\n".join(lines)


def _evidence_section(evidence: list[dict[str, Any]]) -> str:
    lines = ["## 6. 关键证据与引用"]
    if not evidence:
        return "\n".join([*lines, "- 暂无证据。"])
    for item in evidence[:8]:
        lines.append(
            f"- {item.get('policy_id') or 'unknown'} / {item.get('chunk_id') or 'unknown'}：{item.get('text')}"
        )
    return "\n".join(lines)


def _external_evidence_section(external_evidence: list[dict[str, Any]]) -> str:
    lines = ["## 7. 外部证据与 MCP 工具结果"]
    if not external_evidence:
        return "\n".join([*lines, "- 暂无外部证据。"])
    for item in _prioritize_external_evidence(external_evidence)[:10]:
        source = item.get("server_name") or item.get("source_name") or "external"
        tool = item.get("tool_name") or "source"
        title = item.get("title") or item.get("company_name") or item.get("source_org") or "未命名资料"
        url = item.get("source_url") or ""
        summary = item.get("summary") or item.get("text") or item.get("business_evidence") or ""
        detail = f"{source}.{tool}：{title}"
        if url:
            detail += f"（{url}）"
        if summary:
            detail += f"：{_clip(str(summary), 160)}"
        lines.append(f"- {detail}")
    return "\n".join(lines)


def _prioritize_external_evidence(external_evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "cn-financial": [],
        "web-search": [],
        "cninfo": [],
        "other": [],
    }
    for item in external_evidence:
        server = str(item.get("server_name") or item.get("source_name") or "").lower()
        if server in grouped:
            grouped[server].append(item)
        else:
            grouped["other"].append(item)

    output: list[dict[str, Any]] = []
    limits = {"cn-financial": 4, "web-search": 4, "cninfo": 2, "other": 2}
    for server_name, limit in limits.items():
        output.extend(grouped[server_name][:limit])
    for server_name in ("cn-financial", "web-search", "cninfo", "other"):
        output.extend(grouped[server_name][limits[server_name] :])
    return output


def _uncertainty_section(uncertainties: list[str]) -> str:
    lines = ["## 7. 不确定性和风险提示"]
    if not uncertainties:
        return "\n".join([*lines, "- 当前未记录额外不确定性。"])
    return "\n".join([*lines, *_bullets(uncertainties)])


def _bullets(values: list[str]) -> list[str]:
    if not values:
        return ["- 暂无。"]
    return [f"- {value}" for value in values]


def _clip(text: str, max_chars: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1] + "…"
