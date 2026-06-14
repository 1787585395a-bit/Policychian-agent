from __future__ import annotations

import unittest

from policychain.storage import SQLitePolicyStore
from policychain.tools import search_policy
from scripts.ingest_policy_dir import ingest_policy_directory
from tests.helpers import SAMPLE_MANIFEST_PATH, SAMPLE_PDF_PATH, artifact_db_path


class IngestPolicyDirScriptTests(unittest.TestCase):
    def test_ingest_policy_directory_builds_sqlite_store_from_directory(self) -> None:
        db_path = artifact_db_path("policy_dir_ingest")

        result = ingest_policy_directory(
            db_path=db_path,
            source_dir=SAMPLE_PDF_PATH.parent,
            manifest_path=SAMPLE_MANIFEST_PATH,
            reset=True,
        )

        self.assertEqual(result["discovered_file_count"], 1)
        self.assertEqual(result["ingested_count"], 1)
        self.assertEqual(result["policy_count"], 1)
        self.assertGreaterEqual(result["chunk_count"], 3)
        self.assertEqual(result["ingested_policy_ids"], ["POL-2023-NAT-0048"])

        store = SQLitePolicyStore(db_path)
        try:
            results = search_policy(store, "生成式人工智能", top_k=1)
            self.assertEqual(results[0]["policy_id"], "POL-2023-NAT-0048")
        finally:
            store.close()

    def test_ingest_policy_directory_skips_existing_hashes(self) -> None:
        db_path = artifact_db_path("policy_dir_ingest_skip")

        first = ingest_policy_directory(
            db_path=db_path,
            source_dir=SAMPLE_PDF_PATH.parent,
            manifest_path=SAMPLE_MANIFEST_PATH,
            reset=True,
        )
        second = ingest_policy_directory(
            db_path=db_path,
            source_dir=SAMPLE_PDF_PATH.parent,
            manifest_path=SAMPLE_MANIFEST_PATH,
            reset=False,
        )

        self.assertEqual(first["ingested_count"], 1)
        self.assertEqual(second["ingested_count"], 0)
        self.assertEqual(second["skipped_count"], 1)
        self.assertEqual(second["policy_count"], 1)


if __name__ == "__main__":
    unittest.main()
