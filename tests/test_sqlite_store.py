from __future__ import annotations

import unittest

from policychain.ingestion.id_generator import compute_file_hash
from policychain.ingestion.pipeline import ingest_policy_file
from policychain.storage.sqlite_store import SQLitePolicyStore
from tests.helpers import SAMPLE_MANIFEST_PATH, SAMPLE_PDF_PATH, artifact_db_path, build_sample_store


class SQLitePolicyStoreTests(unittest.TestCase):
    def test_upsert_and_read_policy_metadata_document_and_chunks(self) -> None:
        store = build_sample_store()
        try:
            metadata = store.get_policy_metadata(["POL-2023-NAT-0048"])
            document = store.get_policy_document("POL-2023-NAT-0048")
            chunks = store.get_chunks(policy_id="POL-2023-NAT-0048")

            self.assertEqual(len(metadata), 1)
            self.assertEqual(metadata[0].title, "生成式人工智能服务管理暂行办法")
            self.assertEqual(metadata[0].issuing_agencies, ["国家级政府或主管部门（按官方来源）"])
            self.assertIsNotNone(document)
            assert document is not None
            self.assertGreater(document.char_count, 500)
            self.assertGreaterEqual(len(chunks), 3)
            self.assertEqual(chunks[0].policy_id, "POL-2023-NAT-0048")
        finally:
            store.close()

    def test_counts_and_find_policy_by_hash(self) -> None:
        store = build_sample_store()
        try:
            metadata = store.find_policy_by_hash(compute_file_hash(SAMPLE_PDF_PATH))

            self.assertEqual(store.count_policies(), 1)
            self.assertGreaterEqual(store.count_chunks(), 3)
            self.assertIsNotNone(metadata)
            assert metadata is not None
            self.assertEqual(metadata.policy_id, "POL-2023-NAT-0048")
        finally:
            store.close()

    def test_persistent_database_can_be_reopened(self) -> None:
        db_path = artifact_db_path("persistent_store")
        ingested = ingest_policy_file(SAMPLE_PDF_PATH, manifest_path=SAMPLE_MANIFEST_PATH)

        store = SQLitePolicyStore(db_path)
        try:
            store.upsert_ingested_policy(ingested)
            self.assertEqual(store.count_policies(), 1)
        finally:
            store.close()

        reopened = SQLitePolicyStore(db_path)
        try:
            self.assertEqual(reopened.count_policies(), 1)
            self.assertGreaterEqual(reopened.count_chunks(), 3)
            self.assertEqual(reopened.get_policy_metadata(["POL-2023-NAT-0048"])[0].title, "生成式人工智能服务管理暂行办法")
            self.assertGreaterEqual(len(reopened.search_policy("生成式人工智能", top_k=2)), 1)
        finally:
            reopened.close()

    def test_delete_policy_removes_policy_chunks_and_search_results(self) -> None:
        store = build_sample_store()
        try:
            store.delete_policy("POL-2023-NAT-0048")

            self.assertEqual(store.count_policies(), 0)
            self.assertEqual(store.count_chunks(), 0)
            self.assertEqual(store.search_policy("生成式人工智能"), [])
        finally:
            store.close()

    def test_search_policy_returns_relevant_chunks(self) -> None:
        store = build_sample_store()
        try:
            results = store.search_policy("生成式人工智能", top_k=3)

            self.assertGreaterEqual(len(results), 1)
            self.assertEqual(results[0]["policy_id"], "POL-2023-NAT-0048")
            self.assertIn("生成式人工智能", results[0]["matched_text"])
            self.assertGreater(results[0]["score"], 0)
        finally:
            store.close()

    def test_search_policy_can_use_python_fallback_without_fts(self) -> None:
        ingested = ingest_policy_file(SAMPLE_PDF_PATH, manifest_path=SAMPLE_MANIFEST_PATH)
        store = SQLitePolicyStore(":memory:", enable_fts=False)
        try:
            store.upsert_ingested_policy(ingested)
            results = store.search_policy("生成式人工智能", top_k=2)

            self.assertFalse(store.fts_enabled)
            self.assertGreaterEqual(len(results), 1)
            self.assertEqual(results[0]["policy_id"], "POL-2023-NAT-0048")
        finally:
            store.close()

    def test_search_policy_uses_fts_when_available(self) -> None:
        store = build_sample_store()
        try:
            if not store.fts_enabled:
                self.skipTest("SQLite FTS5 trigram tokenizer is not available")

            results = store.search_policy("生成式人工智能", top_k=2)

            self.assertGreaterEqual(len(results), 1)
            self.assertEqual(results[0]["policy_id"], "POL-2023-NAT-0048")
        finally:
            store.close()

    def test_search_policy_filters_by_year(self) -> None:
        store = build_sample_store()
        try:
            matches = store.search_policy("生成式人工智能", filters={"year_from": 2023, "year_to": 2023})
            misses = store.search_policy("生成式人工智能", filters={"year_from": 2024})

            self.assertGreaterEqual(len(matches), 1)
            self.assertEqual(misses, [])
        finally:
            store.close()

    def test_upsert_replaces_existing_chunks_for_same_policy(self) -> None:
        ingested = ingest_policy_file(SAMPLE_PDF_PATH, manifest_path=SAMPLE_MANIFEST_PATH, max_chunk_chars=2000)
        store = SQLitePolicyStore(":memory:")
        try:
            store.upsert_ingested_policy(ingested)
            first_count = len(store.get_chunks(policy_id=ingested.metadata.policy_id))
            store.upsert_ingested_policy(ingested)
            second_count = len(store.get_chunks(policy_id=ingested.metadata.policy_id))

            self.assertEqual(first_count, second_count)
            self.assertGreater(first_count, 0)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
