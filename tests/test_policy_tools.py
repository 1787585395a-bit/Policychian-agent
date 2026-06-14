from __future__ import annotations

import unittest

from policychain.tools import get_policy_metadata, read_policy_content, search_policy
from tests.helpers import build_sample_store


class PolicyToolsTests(unittest.TestCase):
    def test_search_policy_tool_returns_expected_shape(self) -> None:
        store = build_sample_store()
        try:
            results = search_policy(store, query="生成式人工智能", top_k=2)

            self.assertGreaterEqual(len(results), 1)
            self.assertIn("policy_id", results[0])
            self.assertIn("chunk_id", results[0])
            self.assertIn("matched_text", results[0])
            self.assertEqual(results[0]["policy_id"], "POL-2023-NAT-0048")
        finally:
            store.close()

    def test_search_policy_tool_returns_empty_list_for_no_results(self) -> None:
        store = build_sample_store()
        try:
            results = search_policy(store, query="完全不存在的政策关键词", top_k=2)

            self.assertEqual(results, [])
        finally:
            store.close()

    def test_search_policy_tool_keeps_filters(self) -> None:
        store = build_sample_store()
        try:
            matches = search_policy(store, query="生成式人工智能", filters={"year_from": 2023, "year_to": 2023})
            misses = search_policy(store, query="生成式人工智能", filters={"year_from": 2024})

            self.assertGreaterEqual(len(matches), 1)
            self.assertEqual(misses, [])
        finally:
            store.close()

    def test_get_policy_metadata_tool_returns_dicts(self) -> None:
        store = build_sample_store()
        try:
            metadata = get_policy_metadata(store, ["POL-2023-NAT-0048"])

            self.assertEqual(len(metadata), 1)
            self.assertEqual(metadata[0]["title"], "生成式人工智能服务管理暂行办法")
            self.assertEqual(metadata[0]["document_number"], "第15号")
        finally:
            store.close()

    def test_read_policy_content_tool_reads_full_document(self) -> None:
        store = build_sample_store()
        try:
            result = read_policy_content(store, policy_id="POL-2023-NAT-0048")

            self.assertEqual(result["policy_id"], "POL-2023-NAT-0048")
            self.assertIsNotNone(result["metadata"])
            self.assertIsNotNone(result["document"])
            self.assertGreaterEqual(len(result["chunks"]), 3)
        finally:
            store.close()

    def test_read_policy_content_tool_includes_neighbor_chunks(self) -> None:
        store = build_sample_store(max_chunk_chars=500)
        try:
            chunks = store.get_chunks(policy_id="POL-2023-NAT-0048")
            middle_chunk = chunks[1]
            result = read_policy_content(
                store,
                policy_id="POL-2023-NAT-0048",
                chunk_ids=[middle_chunk.chunk_id],
                include_neighbors=True,
            )
            returned_ids = {chunk["chunk_id"] for chunk in result["chunks"]}

            self.assertIsNone(result["document"])
            self.assertIn(middle_chunk.chunk_id, returned_ids)
            self.assertIn(middle_chunk.previous_chunk_id, returned_ids)
            self.assertIn(middle_chunk.next_chunk_id, returned_ids)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
