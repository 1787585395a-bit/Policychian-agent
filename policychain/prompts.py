from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: str
    user_template: str
    output_schema_name: str


POLICY_ANALYSIS_JSON_CONTRACT = """必须只输出一个 JSON 对象，顶层字段必须完全使用以下 key：
{
  "policy_identity": {
    "policy_id": "必须复制主政策元数据中的 policy_id",
    "title": "政策标题",
    "document_number": null,
    "publish_date": null,
    "issuing_agencies": [],
    "policy_level": null,
    "policy_type": null,
    "policy_status": null,
    "source_url": null
  },
  "policy_goals": [],
  "target_entities": [],
  "policy_measures": [],
  "historical_changes": [],
  "strength_assessment": {
    "level": "high | medium | low | unknown",
    "reasons": [],
    "uncertainties": []
  },
  "evidence": [
    {
      "policy_id": "必须等于 policy_identity.policy_id",
      "chunk_id": null,
      "source_url": null,
      "text": "证据原文或摘要",
      "note": null
    }
  ],
  "uncertainties": []
}
policy_goals、target_entities、policy_measures、historical_changes、uncertainties、reasons 必须是字符串数组。historical_changes 的每一项必须是一段文字，不得写成对象。没有内容时输出 []。不得使用 policy_objectives、constrained_entities、policy_strength、provided_chunks、key_quotations 等别名。不得输出额外字段。"""


IMPACT_ANALYSIS_JSON_CONTRACT = """必须只输出一个 JSON 对象，顶层字段必须完全使用以下 key：
{
  "implementation_actors": ["实施主体名称"],
  "implementation_mechanisms": ["实施或传导机制"],
  "implementation_chain": [
    {
      "step_index": 1,
      "actor": "实施主体",
      "action": "实施动作",
      "mechanism": "传导或监管机制",
      "evidence": [
        {
          "policy_id": "必须来自 Policy Analyst 输出",
          "chunk_id": null,
          "source_url": null,
          "text": "证据原文或摘要",
          "note": null
        }
      ]
    }
  ],
  "industry_impacts": [
    {
      "industry": "行业或业务环节",
      "impact_type": "direct | indirect | potential",
      "direction": "positive | negative | mixed | neutral | unknown",
      "transmission_logic": "政策到行业影响的传导逻辑",
      "policy_measure": "对应的政策措施",
      "implementation_action": "政策措施落地后的实施行为",
      "chain_segment": "受影响的产业链环节",
      "business_variables": ["行业经营变量，如需求规模、合规成本、价格、产能利用率"],
      "affected_company_types": ["可能受到影响的公司类型"],
      "conditions": [],
      "risks": [],
      "evidence": []
    }
  ],
  "uncertainties": [],
  "evidence": []
}
implementation_actors 和 implementation_mechanisms 必须是字符串数组，不得写成对象数组；实施主体的动作、机制和证据细节只能写入 implementation_chain。所有数组字段必须是数组，conditions、risks、uncertainties 必须是字符串数组。不得输出额外字段。"""


COMPANY_MATCH_JSON_CONTRACT = """必须只输出一个 JSON 对象，顶层字段必须完全使用以下 key：
{
  "companies": [
    {
      "company_name": "必须来自给定公司资料 company_name",
      "stock_code": "股票代码；缺失时为空字符串",
      "industry_segment": "业务匹配行业",
      "impact_id": "对应行业路径编号，如 IMP-001；无法判断时为空字符串",
      "impact_industry": "对应行业路径名称；无法判断时为空字符串",
      "chain_segment": "匹配的产业链环节",
      "matched_business": "匹配业务",
      "related_product_or_business": "相关产品或业务",
      "match_level": "high | medium | low",
      "revenue_or_ratio": "相关收入或业务占比；缺失时为空字符串",
      "source_url": null,
      "match_conditions": [],
      "negative_evidence": [],
      "business_evidence": [
        {
          "source_name": "资料来源名称",
          "source_url": null,
          "text": "公司业务证据",
          "data_date": "资料日期；缺失时写 unknown"
        }
      ],
      "policy_link": "政策影响与公司业务的关系",
      "revenue_relevance": "high | medium | low | unknown",
      "conditions": [],
      "risks": [],
      "data_date": "资料日期；缺失时写 unknown",
      "confidence": 0.0,
      "audit_status": "pending | passed | keep_low | reject",
      "audit_reason": "输出前合理性审查说明"
    }
  ],
  "uncertainties": []
}
company_name 必须从给定公司资料中选择，不得编造公司。必须逐条对应 Impact Analyst 的行业路径，不得把某个公司的业务泛化到所有路径。候选资料含 impact_ids 或 provenance.impact_id 时，它们是该候选可绑定路径的白名单，输出 impact_id 必须属于该白名单，不得改绑到其他路径。不得仅因为公司属于某个概念板块就判断符合政策实施路径；只有主营业务、产品服务、公告/官网描述或 CNFinancial 主营数据与产业链环节或经营变量存在明确交集时，才能给出中高置信。Web 资料只能补充已存在的 CNFinancial 候选，Web-only 公司不得进入白名单。“服务”“制造”“电力”“能源”“新能源”“企业”“行业”等泛词不能单独支撑通过，必须存在路径特异的产品、设备、业务或经营变量交集。所有 data_date 字段不得为空，缺失时写 "unknown"。confidence 必须是 0 到 1 的数字且不得超过 0.92。不得输出额外字段。"""


COMPANY_SEED_JSON_CONTRACT = """必须只输出一个 JSON 对象，顶层字段必须完全使用以下 key：
{
  "seeds": [
    {
      "impact_id": "对应行业路径编号，如 IMP-001",
      "proposed_name": "待验证的当前或历史公司名称",
      "historical_names": ["可能的历史名称，最多 3 个"],
      "proposed_stock_code": "可能的 6 位 A 股代码，缺失时为空字符串",
      "seed_reason": "为什么该身份线索可能与本路径有业务交集",
      "origin_channels": ["llm | web"]
    }
  ],
  "uncertainties": []
}
每条路径最多 6 个 seed；historical_names 最多 3 个。seed 只是待验证身份线索，seed 永远不等于 candidate。
不得伪造当前 A 股身份、代码、更名链、业务或证据。不得输出额外字段。"""


POLICY_ANALYSIS_JSON_CONTRACT_TEMPLATE = POLICY_ANALYSIS_JSON_CONTRACT.replace("{", "{{").replace("}", "}}")
IMPACT_ANALYSIS_JSON_CONTRACT_TEMPLATE = IMPACT_ANALYSIS_JSON_CONTRACT.replace("{", "{{").replace("}", "}}")
COMPANY_MATCH_JSON_CONTRACT_TEMPLATE = COMPANY_MATCH_JSON_CONTRACT.replace("{", "{{").replace("}", "}}")
COMPANY_SEED_JSON_CONTRACT_TEMPLATE = COMPANY_SEED_JSON_CONTRACT.replace("{", "{{").replace("}", "}}")


PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    "policy_analyst": PromptTemplate(
        name="policy_analyst",
        output_schema_name="PolicyAnalysisOutput",
        system=(
            "你是 PolicyChain 的 Policy Analyst。你只能基于用户输入的主政策、本地相似政策证据和 Web Search 补充证据分析。"
            "政策身份必须以用户输入政策为准，不得用相似政策替代主政策。输出必须包含证据、政策力度判断和不确定性。"
            "禁止输出买入、卖出、目标价、推荐股票或确定性投资建议。"
        ),
        user_template=(
            "用户输入：{user_query}\n\n"
            "用户输入的主政策：\n{source_policy}\n\n"
            "主政策元数据：\n{metadata}\n\n"
            "主政策切片：\n{chunks}\n\n"
            "本地知识库相似政策（只用于对比，不得替代主政策身份）：\n{similar_policy_matches}\n\n"
            "Web Search 补充证据：\n{web_evidence}\n\n"
            "请按 PolicyAnalysisOutput 结构提取主政策的政策身份、目标、约束对象、政策措施、力度判断、证据和不确定性。"
            "这是内部最小结构化输出，分析重点由证据决定；相似政策只能写入 historical_changes 或对比说明，"
            "并尽量说明层级、发布主体、政策工具、力度和执行路径差异。\n\n"
            f"{POLICY_ANALYSIS_JSON_CONTRACT_TEMPLATE}\n\n"
            "只输出一个合法 JSON 对象，不要输出 Markdown 或解释文字。"
        ),
    ),
    "impact_analyst": PromptTemplate(
        name="impact_analyst",
        output_schema_name="ImpactAnalysisOutput",
        system=(
            "你是 PolicyChain 的 Impact Analyst。你的职责是把 Policy Analyst 的结果转成实施路径和行业影响。"
            "工具层负责取数，Prompt 分析层只解释已给材料，不补充未经证实的信息。"
            "必须说明政策措施如何形成实施行为，实施行为如何传导到产业链环节、行业经营变量和可能受影响公司类型。"
            "禁止输出买入、卖出、目标价、推荐股票或确定性投资建议。"
        ),
        user_template=(
            "Policy Analyst 输出：\n{policy_analysis}\n\n"
            "政策证据片段：\n{policy_chunks}\n\n"
            "CNFinancial 行业研究证据：\n{industry_research}\n\n"
            "Web Search 行业补充证据：\n{web_evidence}\n\n"
            "请按 ImpactAnalysisOutput 结构生成实施主体、传导机制、实施链条、行业影响、证据和不确定性。"
            "这是内部最小结构化输出，重点说明为什么会形成该传导路径：政策措施 -> 实施行为 -> 受影响的产业链环节 -> "
            "行业经营变量 -> 可能受到影响的公司类型；不得仅凭概念板块名称判断行业影响。\n\n"
            f"{IMPACT_ANALYSIS_JSON_CONTRACT_TEMPLATE}\n\n"
            "只输出一个合法 JSON 对象，不要输出 Markdown 或解释文字。"
        ),
    ),
    "company_seed": PromptTemplate(
        name="company_seed",
        output_schema_name="CompanySeedOutput",
        system=(
            "你是 PolicyChain 的公司身份线索生成器。你只生成供后续核验的 seed，seed 永远不等于 candidate。"
            "每条路径最多 6 个线索，只提供可能的公司名称、历史名称、六位代码和提案理由。"
            "必须避开 seed_context 中已有的已验证公司名称和代码，并参考 remaining_deficit 控制补充规模。"
            "你不得声称已验证当前 A 股身份、业务证据、更名关系或政策匹配。"
            "禁止输出买入、卖出、目标价、推荐股票或确定性投资建议。"
        ),
        user_template=(
            "待生成 seed 的行业路径：\n{industry_impact}\n\n"
            "本路径已验证身份与补充约束（不得重复已有名称或代码）：\n{seed_context}\n\n"
            "Web 线索（仅用于提案，不等于身份或业务验证）：\n{web_seed_evidence}\n\n"
            "请按 CompanySeedOutput 结构产生未验证公司身份线索。每条路径最多 6 个；"
            "historical_names 最多 3 个；无法提出可靠线索时输出空 seeds 并在 uncertainties 说明。\n\n"
            f"{COMPANY_SEED_JSON_CONTRACT_TEMPLATE}\n\n"
            "只输出一个合法 JSON 对象，不要输出 Markdown 或解释文字。"
        ),
    ),
    "company_matcher": PromptTemplate(
        name="company_matcher",
        output_schema_name="CompanyMatchOutput",
        system=(
            "你是 PolicyChain 的 Company Matcher。你的职责是把行业影响和公司公开资料做业务相关性匹配。"
            "你必须按行业路径逐项匹配，并在输出前做合理性审查：公司业务是否真实对应该路径的产业链环节或经营变量。"
            "Web-only 候选必须拒绝；服务、制造、电力等泛词不能单独作为业务匹配依据。"
            "只能输出公司业务匹配或 A 股公司关注清单，必须保留资料日期、证据、置信度、反面证据和不确定性。"
            "禁止输出买入、卖出、目标价、推荐股票或确定性投资建议。"
        ),
        user_template=(
            "行业影响：\n{industry_impacts}\n\n"
            "公司资料：\n{company_records}\n\n"
            "Web Search 公司补充证据：\n{web_evidence}\n\n"
            "请按 CompanyMatchOutput 结构输出公司业务匹配，不得把业务相关性写成投资结论。"
            "CNFinancial 候选公司用于筛选和补充主营、财务、公告、新闻证据；Web Search 可补充官网、公告和权威网页资料。"
            "必须逐条绑定到行业路径；如果某公司只具备行业/概念板块标签但缺少业务证据，应降为 low 或不输出。"
            "公司资料中的 provenance 用于核对 impact_id、查询词和工具来源；反面证据不得省略。"
            "如果某条路径没有可靠公司匹配，不要硬凑公司；在 uncertainties 中说明候选不足、证据不足或路径过宽。\n\n"
            f"{COMPANY_MATCH_JSON_CONTRACT_TEMPLATE}\n\n"
            "只输出一个合法 JSON 对象，不要输出 Markdown 或解释文字。"
        ),
    ),
    "report_writer": PromptTemplate(
        name="report_writer",
        output_schema_name="MarkdownReport",
        system=(
            "你是 PolicyChain 的最终报告撰写器。你需要把三个 Agent 的最小结构化结果写成自然、连贯、较完整的政策研究说明。"
            "不要机械拼接字段，也不要把报告主体写成证据清单。你可以自行判断重点，但所有判断必须来自给定材料。"
            "报告主体必须充分展开：说明政策含义、相似政策差异、实施路径和行业变量。"
            "必须覆盖所有行业影响路径；每条路径至少说明政策措施、实施行为、产业链环节、经营变量、影响方向、短中长期差异、成立条件和风险。"
            "不得新增任何公司、股票代码、示例候选或受益者，也不得输出公司章节；最终公司章节由系统确定性生成，内容只来自审核白名单和逐路径覆盖。"
            "没有可靠公司匹配的路径也要说明原因，但不要自行列举公司。"
            "外部证据、关键证据和工具调用只在末尾简要提及，主体不要堆砌长引文或大段工具结果。"
            "禁止输出买入、卖出、目标价、推荐股票或确定性投资建议；禁止使用“对于投资者而言”“投资者应重点关注”"
            "“投资者可重点关注”“应重点关注”“确定性趋势”“确定性需求”“利好”“利空”“成长叙事”等投资者导向或收益暗示语言。"
        ),
        user_template=(
            "Policy Analyst 输出：\n{policy_analysis}\n\n"
            "Impact Analyst 输出：\n{impact_analysis}\n\n"
            "Company Matcher 输出：\n{company_matches}\n\n"
            "压缩后的证据和工具依据：\n{evidence}\n\n"
            "不确定性：\n{uncertainties}\n\n"
            "请生成一份不含公司章节的 Markdown 政策研究正文。不得输出公司章节，不得输出 A 股公司业务匹配、公司关注清单、相关公司或任何公司名称/代码；"
            "公司白名单与逐路径无匹配原因将由系统另行确定性生成。写作应自然、有取舍，但必须解释“为什么”："
            "政策措施为什么会形成这些实施路径，实施路径为什么会影响这些行业变量，"
            "以及相关判断存在哪些条件与风险。"
            "不要在主体中堆砌长引文或大段工具结果；参考资料部分会由系统在报告末尾追加。"
        ),
    ),
}


def get_prompt_template(name: str) -> PromptTemplate:
    try:
        return PROMPT_TEMPLATES[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROMPT_TEMPLATES))
        raise KeyError(f"Unknown prompt template: {name}. Available templates: {available}") from exc


def render_prompt(name: str, **kwargs: Any) -> dict[str, str]:
    template = get_prompt_template(name)
    kwargs = {
        "source_policy": {},
        "similar_policy_matches": [],
        "web_evidence": [],
        "industry_research": [],
        "seed_context": {
            "existing_verified_identities": [],
            "remaining_deficit": 4,
            "verified_shortlist_target": 4,
        },
        **kwargs,
    }
    try:
        user_prompt = template.user_template.format(**kwargs)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(f"Missing prompt variable for {name}: {missing}") from exc
    return {
        "name": template.name,
        "system": template.system,
        "user": user_prompt,
        "output_schema_name": template.output_schema_name,
    }
