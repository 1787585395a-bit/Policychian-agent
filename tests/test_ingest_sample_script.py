from __future__ import annotations

import unittest

from policychain.storage import SQLitePolicyStore
from policychain.tools import read_policy_content, search_policy
from scripts.ingest_sample import ingest_sample_database
from tests.helpers import artifact_db_path


class IngestSampleScriptTests(unittest.TestCase):
    def test_ingest_sample_database_builds_repeatable_sqlite_store(self) -> None:
        db_path = artifact_db_path("sample_ingest")

        first = ingest_sample_database(db_path=db_path, reset=True)
        second = ingest_sample_database(db_path=db_path, reset=False)

        self.assertEqual(first["policy_count"], 1)
        self.assertEqual(second["policy_count"], 1)
        self.assertGreaterEqual(second["chunk_count"], 3)
        self.assertIn("POL-2023-NAT-0048", second["policy_ids"])

        store = SQLitePolicyStore(db_path)
        try:
            results = search_policy(store, "生成式人工智能", top_k=1)
            self.assertEqual(results[0]["policy_id"], "POL-2023-NAT-0048")

            content = read_policy_content(
                store,
                policy_id="POL-2023-NAT-0048",
                chunk_ids=[results[0]["chunk_id"]],
                include_neighbors=True,
            )
            self.assertGreaterEqual(len(content["chunks"]), 1)
            self.assertEqual(content["metadata"]["title"], "生成式人工智能服务管理暂行办法")
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
