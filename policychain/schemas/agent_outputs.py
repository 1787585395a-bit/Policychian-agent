from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidenceItem:
    policy_id: str
    chunk_id: str | None
    source_url: str | None
    text: str
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StrengthAssessment:
    level: str
    reasons: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyAnalysisOutput:
    policy_identity: dict[str, Any]
    policy_goals: list[str] = field(default_factory=list)
    target_entities: list[str] = field(default_factory=list)
    policy_measures: list[str] = field(default_factory=list)
    historical_changes: list[str] = field(default_factory=list)
    strength_assessment: StrengthAssessment = field(
        default_factory=lambda: StrengthAssessment(level="unknown")
    )
    evidence: list[EvidenceItem] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_identity": self.policy_identity,
            "policy_goals": list(self.policy_goals),
            "target_entities": list(self.target_entities),
            "policy_measures": list(self.policy_measures),
            "historical_changes": list(self.historical_changes),
            "strength_assessment": self.strength_assessment.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "uncertainties": list(self.uncertainties),
        }


@dataclass
class ImplementationStep:
    step_index: int
    actor: str
    action: str
    mechanism: str
    evidence: list[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "actor": self.actor,
            "action": self.action,
            "mechanism": self.mechanism,
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass
class IndustryImpact:
    industry: str
    impact_type: str
    direction: str
    transmission_logic: str
    policy_measure: str = ""
    implementation_action: str = ""
    chain_segment: str = ""
    business_variables: list[str] = field(default_factory=list)
    affected_company_types: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "industry": self.industry,
            "impact_type": self.impact_type,
            "direction": self.direction,
            "transmission_logic": self.transmission_logic,
            "policy_measure": self.policy_measure,
            "implementation_action": self.implementation_action,
            "chain_segment": self.chain_segment,
            "business_variables": list(self.business_variables),
            "affected_company_types": list(self.affected_company_types),
            "conditions": list(self.conditions),
            "risks": list(self.risks),
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass
class ImpactAnalysisOutput:
    implementation_actors: list[str] = field(default_factory=list)
    implementation_mechanisms: list[str] = field(default_factory=list)
    implementation_chain: list[ImplementationStep] = field(default_factory=list)
    industry_impacts: list[IndustryImpact] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation_actors": list(self.implementation_actors),
            "implementation_mechanisms": list(self.implementation_mechanisms),
            "implementation_chain": [step.to_dict() for step in self.implementation_chain],
            "industry_impacts": [impact.to_dict() for impact in self.industry_impacts],
            "uncertainties": list(self.uncertainties),
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass
class CompanySeed:
    impact_id: str
    proposed_name: str
    historical_names: list[str] = field(default_factory=list)
    proposed_stock_code: str = ""
    seed_reason: str = ""
    origin_channels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "impact_id": self.impact_id,
            "proposed_name": self.proposed_name,
            "historical_names": list(self.historical_names),
            "proposed_stock_code": self.proposed_stock_code,
            "seed_reason": self.seed_reason,
            "origin_channels": list(self.origin_channels),
        }


@dataclass
class CompanySeedOutput:
    seeds: list[CompanySeed] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeds": [seed.to_dict() for seed in self.seeds],
            "uncertainties": list(self.uncertainties),
        }


@dataclass
class CompanyDiscoveryOutput:
    impact_id: str
    web_queries: list[str] = field(default_factory=list)
    seeds: list[CompanySeed] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "impact_id": self.impact_id,
            "web_queries": list(self.web_queries),
            "seeds": [seed.to_dict() for seed in self.seeds],
            "uncertainties": list(self.uncertainties),
        }


@dataclass
class CompanyEvidence:
    source_name: str
    source_url: str | None
    text: str
    data_date: str
    revenue_or_ratio: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompanyMatch:
    company_name: str
    industry_segment: str
    matched_business: str
    match_level: str
    impact_id: str = ""
    impact_industry: str = ""
    stock_code: str = ""
    chain_segment: str = ""
    related_product_or_business: str = ""
    revenue_or_ratio: str = ""
    source_url: str | None = None
    match_conditions: list[str] = field(default_factory=list)
    negative_evidence: list[str] = field(default_factory=list)
    business_evidence: list[CompanyEvidence] = field(default_factory=list)
    policy_link: str = ""
    revenue_relevance: str = "unknown"
    conditions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    data_date: str = ""
    confidence: float = 0.0
    audit_status: str = "passed"
    audit_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_name": self.company_name,
            "industry_segment": self.industry_segment,
            "matched_business": self.matched_business,
            "match_level": self.match_level,
            "impact_id": self.impact_id,
            "impact_industry": self.impact_industry,
            "stock_code": self.stock_code,
            "chain_segment": self.chain_segment,
            "related_product_or_business": self.related_product_or_business,
            "revenue_or_ratio": self.revenue_or_ratio,
            "source_url": self.source_url,
            "match_conditions": list(self.match_conditions),
            "negative_evidence": list(self.negative_evidence),
            "business_evidence": [item.to_dict() for item in self.business_evidence],
            "policy_link": self.policy_link,
            "revenue_relevance": self.revenue_relevance,
            "conditions": list(self.conditions),
            "risks": list(self.risks),
            "data_date": self.data_date,
            "confidence": self.confidence,
            "audit_status": self.audit_status,
            "audit_reason": self.audit_reason,
        }


@dataclass
class CompanyMatchOutput:
    companies: list[CompanyMatch] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "companies": [company.to_dict() for company in self.companies],
            "uncertainties": list(self.uncertainties),
        }
