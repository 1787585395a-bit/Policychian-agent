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
    "policy_id": "必须复制政策元数据中的 policy_id",
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
policy_goals、target_entities、policy_measures、historical_changes、uncertainties、reasons 必须是字符串数组；historical_changes 的每一项必须是一段文字，不得写成对象。没有内容时输出 []。不得使用 policy_objectives、constrained_entities、policy_strength、provided_chunks、key_quotations 等别名。不要输出额外字段。"""


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
implementation_actors 和 implementation_mechanisms 必须是字符串数组，不得写成对象数组；实施主体的动作、机制和证据细节只能写入 implementation_chain。所有数组字段必须是数组；conditions、risks、uncertainties 必须是字符串数组。不得输出额外字段。"""


COMPANY_MATCH_JSON_CONTRACT = """必须只输出一个 JSON 对象，顶层字段必须完全使用以下 key：
{
  "companies": [
    {
      "company_name": "必须来自给定公司资料 company_name",
      "stock_code": "股票代码，缺失时为空字符串",
      "industry_segment": "业务匹配行业",
      "chain_segment": "匹配的产业链环节",
      "matched_business": "匹配业务",
      "related_product_or_business": "相关产品或业务",
      "match_level": "high | medium | low",
      "annual_report_evidence": [
        {
          "source_name": "最近两期年报名称",
          "source_url": null,
          "text": "年报中的业务证据；没有充分证据时写明未找到充分证据",
          "data_date": "资料年份；缺失时写 unknown",
          "report_year": null,
          "revenue_or_ratio": "",
          "evidence_found": true
        }
      ],
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
      "confidence": 0.0
    }
  ],
  "uncertainties": []
}
company_name 必须从给定公司资料中选择，不得编造公司。不得仅因为公司属于某个概念板块就判断符合政策实施路径。若最近两期年报没有充分证据，match_level 必须为 low，并在 negative_evidence 中写明“未在最近两期年报中找到充分证据”。所有 data_date 字段不得为空，缺失时写 "unknown"。confidence 必须是 0 到 1 的数字。不得输出额外字段。"""


POLICY_ANALYSIS_JSON_CONTRACT_TEMPLATE = POLICY_ANALYSIS_JSON_CONTRACT.replace("{", "{{").replace("}", "}}")
IMPACT_ANALYSIS_JSON_CONTRACT_TEMPLATE = IMPACT_ANALYSIS_JSON_CONTRACT.replace("{", "{{").replace("}", "}}")
COMPANY_MATCH_JSON_CONTRACT_TEMPLATE = COMPANY_MATCH_JSON_CONTRACT.replace("{", "{{").replace("}", "}}")


PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    "policy_analyst": PromptTemplate(
        name="policy_analyst",
        output_schema_name="PolicyAnalysisOutput",
        system=(
            "你是 PolicyChain 的 Policy Analyst。只基于工具层返回的政策元数据和政策片段分析，"
            "不得自行检索或编造缺失信息。输出必须包含证据、政策力度判断和不确定性。"
            "禁止输出买入、卖出、目标价、推荐股票或确定性投资建议。"
        ),
        user_template=(
            "用户问题：{user_query}\n\n"
            "政策元数据：\n{metadata}\n\n"
            "政策片段：\n{chunks}\n\n"
            "Web Search 补充证据：\n{web_evidence}\n\n"
            "请按 PolicyAnalysisOutput 结构提取政策身份、目标、约束对象、政策措施、力度判断、证据和不确定性。\n\n"
            f"{POLICY_ANALYSIS_JSON_CONTRACT_TEMPLATE}\n\n"
            "只输出一个合法 JSON 对象，不要输出 Markdown 或解释文字。"
        ),
    ),
    "impact_analyst": PromptTemplate(
        name="impact_analyst",
        output_schema_name="ImpactAnalysisOutput",
        system=(
            "你是 PolicyChain 的 Impact Analyst。你的职责是把 Policy Analyst 的结构化输出转成实施路径和行业影响。"
            "工具层负责取数，Prompt 分析层只解释已给材料，不补充未经证实的信息。"
            "禁止输出买入、卖出、目标价、推荐股票或确定性投资建议。"
        ),
        user_template=(
            "Policy Analyst 输出：\n{policy_analysis}\n\n"
            "政策证据片段：\n{policy_chunks}\n\n"
            "CNFinancial 行业研究证据：\n{industry_research}\n\n"
            "Web Search 行业补充证据：\n{web_evidence}\n\n"
            "请按 ImpactAnalysisOutput 结构生成实施主体、传导机制、实施链条、行业影响、证据和不确定性。行业影响必须明确：政策措施 -> 实施行为 -> 受影响的产业链环节 -> 行业经营变量 -> 可能受到影响的公司类型；不得仅凭概念板块名称判断行业影响。\n\n"
            f"{IMPACT_ANALYSIS_JSON_CONTRACT_TEMPLATE}\n\n"
            "只输出一个合法 JSON 对象，不要输出 Markdown 或解释文字。"
        ),
    ),
    "company_matcher": PromptTemplate(
        name="company_matcher",
        output_schema_name="CompanyMatchOutput",
        system=(
            "你是 PolicyChain 的 Company Matcher。你的职责是把行业影响和公司公开资料做业务相关性匹配，"
            "只能输出公司业务匹配或公司关注清单，必须保留资料日期、证据、置信度和不确定性。"
            "禁止输出买入、卖出、目标价、推荐股票或确定性投资建议。"
        ),
        user_template=(
            "行业影响：\n{industry_impacts}\n\n"
            "公司资料：\n{company_records}\n\n"
            "最近两期年报证据：\n{annual_report_evidence}\n\n"
            "Web Search 公司补充证据：\n{web_evidence}\n\n"
            "请按 CompanyMatchOutput 结构输出公司业务匹配，不得把业务相关性写成投资结论。CNFinancial 候选公司只用于筛选，最终业务真实性必须优先由最近两期年报原文验证。\n\n"
            f"{COMPANY_MATCH_JSON_CONTRACT_TEMPLATE}\n\n"
            "只输出一个合法 JSON 对象，不要输出 Markdown 或解释文字。"
        ),
    ),
    "report_writer": PromptTemplate(
        name="report_writer",
        output_schema_name="MarkdownReport",
        system=(
            "你是 PolicyChain 的报告撰写器。你只能整合三个 Agent 的结构化结果，"
            "不得新增未被证据支持的政策、行业或公司结论。报告必须显示证据、不确定性和非投资建议边界。"
            "禁止输出买入、卖出、目标价、推荐股票或确定性投资建议。"
        ),
        user_template=(
            "Policy Analyst 输出：\n{policy_analysis}\n\n"
            "Impact Analyst 输出：\n{impact_analysis}\n\n"
            "Company Matcher 输出：\n{company_matches}\n\n"
            "证据：\n{evidence}\n\n"
            "不确定性：\n{uncertainties}\n\n"
            "请生成一份 Markdown 政策研究报告。"
        ),
    ),
}


PROMPT_TEMPLATES["policy_analyst"] = PromptTemplate(
    name="policy_analyst",
    output_schema_name="PolicyAnalysisOutput",
    system=(
        "\u4f60\u662f PolicyChain \u7684 Policy Analyst\u3002\u4f60\u53ea\u80fd\u57fa\u4e8e\u7528\u6237\u8f93\u5165\u7684\u4e3b\u653f\u7b56\u3001"
        "\u672c\u5730\u76f8\u4f3c\u653f\u7b56\u8bc1\u636e\u548c Web Search \u8865\u5145\u8bc1\u636e\u5206\u6790\u3002"
        "\u8f93\u51fa\u5fc5\u987b\u5305\u542b\u8bc1\u636e\u3001\u653f\u7b56\u529b\u5ea6\u5224\u65ad\u548c\u4e0d\u786e\u5b9a\u6027\u3002"
        "\u653f\u7b56\u8eab\u4efd\u5fc5\u987b\u4ee5\u7528\u6237\u8f93\u5165\u7684\u4e3b\u653f\u7b56\u4e3a\u51c6\uff0c\u4e0d\u5f97\u7528\u76f8\u4f3c\u653f\u7b56\u66ff\u4ee3\u4e3b\u653f\u7b56\u3002"
        "\u7981\u6b62\u8f93\u51fa\u4e70\u5165\u3001\u5356\u51fa\u3001\u76ee\u6807\u4ef7\u3001\u63a8\u8350\u80a1\u7968\u6216\u786e\u5b9a\u6027\u6295\u8d44\u5efa\u8bae\u3002"
    ),
    user_template=(
        "\u7528\u6237\u8f93\u5165\uff1a{user_query}\n\n"
        "\u7528\u6237\u8f93\u5165\u7684\u4e3b\u653f\u7b56\uff1a\n{source_policy}\n\n"
        "\u4e3b\u653f\u7b56\u5143\u6570\u636e\uff1a\n{metadata}\n\n"
        "\u4e3b\u653f\u7b56\u5207\u7247\uff1a\n{chunks}\n\n"
        "\u672c\u5730\u77e5\u8bc6\u5e93\u76f8\u4f3c\u653f\u7b56\uff08\u53ea\u7528\u4e8e\u5bf9\u6bd4\uff0c\u4e0d\u5f97\u66ff\u4ee3\u4e3b\u653f\u7b56\u8eab\u4efd\uff09\uff1a\n{similar_policy_matches}\n\n"
        "Web Search \u8865\u5145\u8bc1\u636e\uff1a\n{web_evidence}\n\n"
        "\u8bf7\u6309 PolicyAnalysisOutput \u7ed3\u6784\u63d0\u53d6\u4e3b\u653f\u7b56\u7684\u653f\u7b56\u8eab\u4efd\u3001\u76ee\u6807\u3001\u7ea6\u675f\u5bf9\u8c61\u3001\u653f\u7b56\u63aa\u65bd\u3001\u529b\u5ea6\u5224\u65ad\u3001\u8bc1\u636e\u548c\u4e0d\u786e\u5b9a\u6027\u3002"
        "\u76f8\u4f3c\u653f\u7b56\u53ea\u80fd\u5199\u5165 historical_changes \u6216\u5bf9\u6bd4\u8bf4\u660e\u3002\n\n"
        f"{POLICY_ANALYSIS_JSON_CONTRACT_TEMPLATE}\n\n"
        "\u53ea\u8f93\u51fa\u4e00\u4e2a\u5408\u6cd5 JSON \u5bf9\u8c61\uff0c\u4e0d\u8981\u8f93\u51fa Markdown \u6216\u89e3\u91ca\u6587\u5b57\u3002"
    ),
)


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
        "annual_report_evidence": [],
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
