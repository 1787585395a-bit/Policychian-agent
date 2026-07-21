from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolicyResearchState:
    user_query: str
    source_policy: dict[str, Any] = field(default_factory=dict)
    similar_policy_matches: list[dict[str, Any]] = field(default_factory=list)
    policy_ids: list[str] = field(default_factory=list)
    policy_documents: list[dict[str, Any]] = field(default_factory=list)
    policy_chunks: list[dict[str, Any]] = field(default_factory=list)
    policy_analysis: dict[str, Any] = field(default_factory=dict)
    implementation_path: list[dict[str, Any]] = field(default_factory=list)
    industry_impacts: list[dict[str, Any]] = field(default_factory=list)
    company_candidates: list[dict[str, Any]] = field(default_factory=list)
    company_matches: list[dict[str, Any]] = field(default_factory=list)
    company_coverage: list[dict[str, Any]] = field(default_factory=list)
    company_match_audit: list[dict[str, Any]] = field(default_factory=list)
    react_candidate_audit: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    external_evidence: list[dict[str, Any]] = field(default_factory=list)
    policy_web_evidence: list[dict[str, Any]] = field(default_factory=list)
    industry_research: list[dict[str, Any]] = field(default_factory=list)
    company_research: list[dict[str, Any]] = field(default_factory=list)
    react_traces: list[dict[str, Any]] = field(default_factory=list)
    tool_call_logs: list[dict[str, Any]] = field(default_factory=list)
    progress_events: list[dict[str, Any]] = field(default_factory=list)
    run_id: str = ""
    run_mode: str = "deterministic"
    agent_status: dict[str, str] = field(default_factory=dict)
    fallback_used: bool = False
    uncertainties: list[str] = field(default_factory=list)
    final_report: str = ""
