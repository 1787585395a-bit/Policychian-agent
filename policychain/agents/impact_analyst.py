from __future__ import annotations

import re
from typing import Any, Iterable

from policychain.schemas.agent_outputs import (
    EvidenceItem,
    ImpactAnalysisOutput,
    ImplementationStep,
    IndustryImpact,
)
from policychain.mcp import MCPToolInvoker, consume_mcp_invoker_errors, is_unavailable_invoker
from policychain.safety import assert_no_investment_advice
from policychain.state import PolicyResearchState
from policychain.tools import collect_impact_research
from policychain.tools.mcp_tools import mcp_unavailable_uncertainty


ACTOR_MARKERS = (
    "生成式人工智能服务提供者",
    "服务提供者",
    "提供者",
    "主管部门",
    "国家",
    "行业组织",
    "用户",
    "未成年人",
)
MECHANISM_RULES = (
    ("备案", "备案与合规审查"),
    ("安全评估", "安全评估与风险处置"),
    ("监督", "主管部门监督管理"),
    ("投诉", "投诉受理和处置机制"),
    ("举报", "投诉受理和处置机制"),
    ("个人信息", "数据与个人信息保护"),
    ("训练数据", "训练数据治理"),
    ("标识", "生成内容标识管理"),
    ("算法", "算法模型治理"),
    ("模型", "算法模型治理"),
)
INDUSTRY_RULES = (
    ("生成式人工智能服务", ("生成式人工智能", "服务提供者"), "direct", "mixed", "政策直接规范生成式人工智能服务提供、内容安全和合规责任。"),
    ("算法模型研发与评估", ("算法", "模型", "训练数据"), "direct", "mixed", "政策要求模型、算法和训练数据环节承担安全治理责任。"),
    ("数据治理与安全合规", ("数据", "个人信息", "安全评估"), "indirect", "mixed", "政策通过数据来源、个人信息和安全评估要求传导到数据治理能力建设。"),
    ("互联网内容服务", ("网络信息", "内容", "标识"), "direct", "mixed", "政策把生成内容纳入信息内容治理和标识管理要求。"),
    ("未成年人保护相关服务", ("未成年人",), "potential", "mixed", "涉及未成年人使用场景时，服务设计和风险防护要求可能提高。"),
)


class ImpactAnalysisError(RuntimeError):
    """Raised when Impact Analyst cannot produce a structured result."""


def run_impact_analyst(
    state: PolicyResearchState,
    mcp_invoker: MCPToolInvoker | None = None,
) -> ImpactAnalysisOutput:
    """Run a deterministic Impact Analyst pass over Policy Analyst output."""

    if not state.policy_analysis or state.policy_analysis.get("policy_identity", {}).get("status") == "no_policy_found":
        output = ImpactAnalysisOutput(
            uncertainties=["缺少可用的 Policy Analyst 输出，无法生成实施路径和行业影响分析。"]
        )
        _write_state(state, output)
        return output

    output = analyze_policy_impact(
        policy_analysis=state.policy_analysis,
        policy_chunks=state.policy_chunks,
    )
    research = collect_impact_research(
        policy_analysis=state.policy_analysis,
        industry_impacts=[impact.to_dict() for impact in output.industry_impacts],
        invoker=mcp_invoker,
    )
    state.industry_research = [*research["cnfinancial"], *research["web"]]
    state.tool_call_logs.extend(research.get("tool_logs", []))
    state.external_evidence = _merge_external_evidence(state.external_evidence, state.industry_research)
    if is_unavailable_invoker(mcp_invoker):
        state.uncertainties = _unique(
            [
                *state.uncertainties,
                mcp_unavailable_uncertainty("CNFinancial"),
                mcp_unavailable_uncertainty("Open-WebSearch"),
            ]
        )
    state.uncertainties = _unique([*state.uncertainties, *consume_mcp_invoker_errors(mcp_invoker)])
    _write_state(state, output)
    return output


def analyze_policy_impact(
    policy_analysis: dict[str, Any],
    policy_chunks: list[dict[str, Any]],
) -> ImpactAnalysisOutput:
    if not policy_analysis:
        raise ImpactAnalysisError("Impact Analyst requires policy_analysis")

    text = _analysis_text(policy_analysis, policy_chunks)
    policy_id = str(policy_analysis.get("policy_identity", {}).get("policy_id") or "")
    source_url = policy_analysis.get("policy_identity", {}).get("source_url")
    actors = _extract_actors(policy_analysis, text)
    mechanisms = _extract_mechanisms(text)
    evidence = _build_evidence(policy_id=policy_id, source_url=source_url, chunks=policy_chunks)
    implementation_chain = _build_implementation_chain(
        policy_analysis=policy_analysis,
        actors=actors,
        mechanisms=mechanisms,
        evidence=evidence,
    )
    industry_impacts = _build_industry_impacts(text=text, evidence=evidence)
    uncertainties = _uncertainties(implementation_chain, industry_impacts)

    output = ImpactAnalysisOutput(
        implementation_actors=actors,
        implementation_mechanisms=mechanisms,
        implementation_chain=implementation_chain,
        industry_impacts=industry_impacts,
        uncertainties=uncertainties,
        evidence=evidence,
    )
    assert_no_investment_advice(output.to_dict(), context="Impact analysis output")
    return output


def _write_state(state: PolicyResearchState, output: ImpactAnalysisOutput) -> None:
    payload = output.to_dict()
    state.implementation_path = payload["implementation_chain"]
    state.industry_impacts = payload["industry_impacts"]
    state.evidence = _merge_evidence(state.evidence, payload["evidence"])
    state.uncertainties = _unique([*state.uncertainties, *payload["uncertainties"]])


def _analysis_text(policy_analysis: dict[str, Any], chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for key in ("policy_goals", "target_entities", "policy_measures"):
        values = policy_analysis.get(key) or []
        parts.extend(str(value) for value in values)
    parts.extend(str(chunk.get("content") or "") for chunk in chunks)
    return "\n".join(parts)


def _extract_actors(policy_analysis: dict[str, Any], text: str) -> list[str]:
    actors = list(policy_analysis.get("target_entities") or [])
    actors.extend(marker for marker in ACTOR_MARKERS if marker in text)
    actors = _unique(actors)
    if "提供者" in actors and "生成式人工智能服务提供者" in actors:
        actors.remove("提供者")
    return actors[:6]


def _extract_mechanisms(text: str) -> list[str]:
    mechanisms = [label for marker, label in MECHANISM_RULES if marker in text]
    return _unique(mechanisms)[:6]


def _build_implementation_chain(
    policy_analysis: dict[str, Any],
    actors: list[str],
    mechanisms: list[str],
    evidence: list[EvidenceItem],
    limit: int = 5,
) -> list[ImplementationStep]:
    measures = list(policy_analysis.get("policy_measures") or [])
    if not measures:
        return []

    steps: list[ImplementationStep] = []
    for measure in measures:
        actor = _choose_actor(measure, actors)
        mechanism = _choose_mechanism(measure, mechanisms)
        steps.append(
            ImplementationStep(
                step_index=len(steps) + 1,
                actor=actor,
                action=_clip(str(measure), max_chars=180),
                mechanism=mechanism,
                evidence=evidence[:1],
            )
        )
        if len(steps) >= limit:
            break
    return steps


def _choose_actor(measure: str, actors: list[str]) -> str:
    for actor in actors:
        if actor and actor in measure:
            return actor
    if "监督" in measure or "管理" in measure:
        return "主管部门"
    return actors[0] if actors else "政策实施相关主体"


def _choose_mechanism(measure: str, mechanisms: list[str]) -> str:
    for marker, label in MECHANISM_RULES:
        if marker in measure:
            return label
    return mechanisms[0] if mechanisms else "政策要求传导"


def _build_industry_impacts(
    text: str,
    evidence: list[EvidenceItem],
    limit: int = 5,
) -> list[IndustryImpact]:
    impacts: list[IndustryImpact] = []
    for industry, markers, impact_type, direction, logic in INDUSTRY_RULES:
        if any(marker in text for marker in markers):
            impacts.append(
                IndustryImpact(
                    industry=industry,
                    impact_type=impact_type,
                    direction=direction,
                    transmission_logic=logic,
                    policy_measure=_clip(logic, max_chars=120),
                    implementation_action=_clip(logic, max_chars=120),
                    chain_segment=industry,
                    business_variables=_business_variables_for_industry(industry),
                    affected_company_types=_company_types_for_industry(industry),
                    conditions=["需结合监管执行口径、配套细则和具体业务场景判断影响强度。"],
                    risks=["若企业缺少内容安全、数据治理或模型评估能力，合规成本和整改压力可能上升。"],
                    evidence=evidence[:1],
                )
            )
        if len(impacts) >= limit:
            break
    return impacts


def _build_evidence(
    policy_id: str,
    source_url: str | None,
    chunks: list[dict[str, Any]],
    limit: int = 5,
) -> list[EvidenceItem]:
    evidence: list[EvidenceItem] = []
    for chunk in chunks:
        content = _compact(str(chunk.get("content") or ""))
        if not content:
            continue
        evidence.append(
            EvidenceItem(
                policy_id=policy_id,
                chunk_id=chunk.get("chunk_id"),
                source_url=source_url,
                text=_clip(content, max_chars=220),
                note=chunk.get("section_title"),
            )
        )
        if len(evidence) >= limit:
            break
    return evidence


def _uncertainties(
    implementation_chain: list[ImplementationStep],
    industry_impacts: list[IndustryImpact],
) -> list[str]:
    uncertainties = ["当前行业影响为基于单份政策文本的一级/二级传导分析，尚未接入产业数据和后续配套政策。"]
    if not implementation_chain:
        uncertainties.append("未从 Policy Analyst 输出中稳定形成实施链条。")
    if not industry_impacts:
        uncertainties.append("未识别出可归类的行业影响。")
    return uncertainties


def _merge_evidence(existing: list[dict[str, Any]], new_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None]] = set()
    for item in [*existing, *new_items]:
        key = (item.get("policy_id"), item.get("chunk_id"))
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _merge_external_evidence(existing: list[dict[str, Any]], new_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in [*existing, *new_items]:
        key = (
            str(item.get("server_name") or ""),
            str(item.get("tool_name") or ""),
            str(item.get("source_url") or item.get("title") or item.get("query") or ""),
        )
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _business_variables_for_industry(industry: str) -> list[str]:
    if "数据" in industry or "安全" in industry:
        return ["合规成本", "数据质量", "安全评估需求", "客户付费意愿"]
    if "模型" in industry or "算法" in industry:
        return ["研发投入", "评测需求", "训练数据质量", "备案和安全评估成本"]
    if "内容" in industry:
        return ["内容审核成本", "标识系统需求", "平台治理投入"]
    return ["需求规模", "合规成本", "产品价格", "产能利用率"]


def _company_types_for_industry(industry: str) -> list[str]:
    if "数据" in industry or "安全" in industry:
        return ["数据治理服务商", "网络安全服务商", "合规审计服务商"]
    if "模型" in industry or "算法" in industry:
        return ["大模型服务商", "模型评测机构", "AI 平台服务商"]
    if "内容" in industry:
        return ["内容审核工具提供商", "互联网平台治理服务商"]
    return ["位于该产业链环节且有公开业务证据的公司"]


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        compacted = _compact(str(value))
        if compacted and compacted not in seen:
            seen.add(compacted)
            output.append(compacted)
    return output


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clip(text: str, max_chars: int = 180) -> str:
    compacted = _compact(text)
    if len(compacted) <= max_chars:
        return compacted
    return compacted[: max_chars - 1].rstrip() + "…"
