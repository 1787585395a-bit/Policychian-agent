from __future__ import annotations

import unittest
from pathlib import Path


class AgentsRulesTests(unittest.TestCase):
    def test_agents_md_documents_current_product_flow_and_review_rules(self) -> None:
        text = Path("AGENTS.md").read_text(encoding="utf-8")

        required_phrases = (
            "用户输入政策链接或政策正文",
            "本地政策知识库只用于查找相似政策",
            "Policy Analyst",
            "Impact Analyst",
            "Company Matcher",
            "A 股公司业务匹配",
            "不得包含买入、卖出、目标价",
            "前端审查规则",
            "日志审查规则",
            "不得等待进程退出",
            "OwningProcess",
            "正文质量校验",
        )
        for phrase in required_phrases:
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
