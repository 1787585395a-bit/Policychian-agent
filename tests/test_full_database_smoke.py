from __future__ import annotations

import unittest

from policychain.paths import FULL_DB_PATH
from policychain.safety import assert_no_investment_advice
from policychain.storage import SQLitePolicyStore
from scripts.run_research import run_research


class FullDatabaseSmokeTests(unittest.TestCase):
    @unittest.skipUnless(FULL_DB_PATH.exists(), "Full policy database has not been built locally.")
    def test_full_database_runs_deterministic_workflow(self) -> None:
        store = SQLitePolicyStore(FULL_DB_PATH)
        try:
            self.assertGreaterEqual(store.count_policies(), 50)
            self.assertGreaterEqual(store.count_chunks(), 300)
        finally:
            store.close()

        report = run_research(
            query="生成式人工智能服务提供者有哪些管理要求",
            db_path=FULL_DB_PATH,
            ensure_sample_db=True,
            use_llm=False,
        )

        self.assertIn("PolicyChain 政策研究报告", report)
        self.assertIn("生成式人工智能服务管理暂行办法", report)
        assert_no_investment_advice(report, context="full database smoke report")


if __name__ == "__main__":
    unittest.main()
