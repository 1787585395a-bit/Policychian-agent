from __future__ import annotations

import unittest

from policychain.tools import load_mock_companies, read_company_source, search_company_information


class CompanyToolsTests(unittest.TestCase):
    def test_load_mock_companies(self) -> None:
        companies = load_mock_companies()

        self.assertGreaterEqual(len(companies), 1)
        self.assertIn("company_name", companies[0])

    def test_search_company_information_matches_industry(self) -> None:
        results = search_company_information("生成式人工智能服务", keywords=["生成式人工智能"], top_k=2)

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["industry_segment"], "生成式人工智能服务")

    def test_read_company_source_returns_evidence_fields(self) -> None:
        company = search_company_information("数据治理与安全合规", keywords=["数据"], top_k=1)[0]
        source = read_company_source(company)

        self.assertTrue(source["source_name"])
        self.assertTrue(source["text"])
        self.assertTrue(source["data_date"])


if __name__ == "__main__":
    unittest.main()
