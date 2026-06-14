from __future__ import annotations

import json
from typing import Iterable

from policychain.llm import LLMClient, create_llm_client
from policychain.mcp import MCPToolInvoker, consume_mcp_invoker_errors, is_unavailable_invoker
from policychain.prompts import render_prompt
from policychain.schemas.agent_outputs import ImpactAnalysisOutput
from policychain.state import PolicyResearchState
from policychain.structured_output import parse_structured_output
from policychain.tools import collect_impact_research
from policychain.tools.mcp_tools import mcp_unavailable_uncertainty


class LLMImpactAnalysisError(RuntimeError):
    """Raised when LLM Impact Analyst output cannot be trusted."""


def run_llm_impact_analyst(
    state: PolicyResearchState,
    llm_client: LLMClient | None = None,
    mcp_invoker: MCPToolInvoker | None = None,
) -> ImpactAnalysisOutput:
    """Run the optional LLM Impact Analyst over Policy Analyst output."""

    if not state.policy_analysis or state.policy_analysis.get("policy_identity", {}).get("status") == "no_policy_found":
        output = ImpactAnalysisOutput(
            uncertainties=["缺少可用的 Policy Analyst 输出，无法生成实施路径和行业影响分析。"]
        )
        _write_state(state, output)
        return output

    research = collect_impact_research(
        policy_analysis=state.policy_analysis,
        industry_impacts=state.industry_impacts,
        invoker=mcp_invoker,
    )
    state.industry_research = [*research["cnfinancial"], *research["web"]]
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

    prompt = render_prompt(
        "impact_analyst",
        policy_analysis=_json_for_prompt(state.policy_analysis),
        policy_chunks=_json_for_prompt(state.policy_chunks),
        industry_research=_json_for_prompt(research["cnfinancial"]),
        web_evidence=_json_for_prompt(research["web"]),
    )
    client = llm_client or create_llm_client()
    raw_output = client.generate(prompt["system"], prompt["user"])
    output = parse_structured_output(raw_output, prompt["output_schema_name"])
    if not isinstance(output, ImpactAnalysisOutput):
        raise LLMImpactAnalysisError("LLM Impact Analyst returned an unexpected output schema")
    _assert_evidence_policy_ids(output, _allowed_policy_ids(state))
    _write_state(state, output)
    return output


def _write_state(state: PolicyResearchState, output: ImpactAnalysisOutput) -> None:
    payload = output.to_dict()
    state.implementation_path = payload["implementation_chain"]
    state.industry_impacts = payload["industry_impacts"]
    state.evidence = _merge_evidence(state.evidence, payload["evidence"])
    state.uncertainties = _unique([*state.uncertainties, *payload["uncertainties"]])


def _assert_evidence_policy_ids(output: ImpactAnalysisOutput, allowed_policy_ids: set[str]) -> None:
    evidence_policy_ids: set[str] = set()
    evidence_policy_ids.update(item.policy_id for item in output.evidence)
    for step in output.implementation_chain:
        evidence_policy_ids.update(item.policy_id for item in step.evidence)
    for impact in output.industry_impacts:
        evidence_policy_ids.update(item.policy_id for item in impact.evidence)

    mismatches = sorted(policy_id for policy_id in evidence_policy_ids if policy_id not in allowed_policy_ids)
    if mismatches:
        raise LLMImpactAnalysisError(
            f"LLM Impact Analyst evidence policy_id mismatch: {', '.join(mismatches)}"
        )


def _allowed_policy_ids(state: PolicyResearchState) -> set[str]:
    policy_ids = set(state.policy_ids)
    identity_policy_id = str(state.policy_analysis.get("policy_identity", {}).get("policy_id") or "")
    if identity_policy_id:
        policy_ids.add(identity_policy_id)
    if not policy_ids:
        raise LLMImpactAnalysisError("LLM Impact Analyst requires policy_ids in state")
    return policy_ids


def _json_for_prompt(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _merge_evidence(existing: list[dict[str, object]], new_items: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    seen: set[tuple[object, object]] = set()
    for item in [*existing, *new_items]:
        key = (item.get("policy_id"), item.get("chunk_id"))
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _merge_external_evidence(existing: list[dict[str, object]], new_items: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    seen: set[tuple[object, object, object]] = set()
    for item in [*existing, *new_items]:
        key = (item.get("server_name"), item.get("tool_name"), item.get("source_url") or item.get("title") or item.get("query"))
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output
