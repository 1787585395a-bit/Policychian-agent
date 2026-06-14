from __future__ import annotations

import unittest

from policychain.ingestion.pipeline import ingest_policy_file
from policychain.schemas.policy_schema import IngestedPolicy
from tests.helpers import SAMPLE_MANIFEST_PATH, SAMPLE_PDF_PATH


class IngestionTests(unittest.TestCase):
    def test_ingest_real_sample_policy(self) -> None:
        result = ingest_policy_file(
            SAMPLE_PDF_PATH,
            manifest_path=SAMPLE_MANIFEST_PATH,
            max_chunk_chars=1200,
        )

        self.assertIsInstance(result, IngestedPolicy)
        self.assertEqual(result.metadata.policy_id, "POL-2023-NAT-0048")
        self.assertEqual(result.document.policy_id, result.metadata.policy_id)
        self.assertEqual(result.document.file_hash, result.metadata.file_hash)
        self.assertGreater(result.document.char_count, 500)
        self.assertGreaterEqual(len(result.chunks), 3)
        self.assertIsNone(result.chunks[0].previous_chunk_id)
        self.assertIsNone(result.chunks[-1].next_chunk_id)
        self.assertTrue(any("生成式人工智能" in chunk.content for chunk in result.chunks))


if __name__ == "__main__":
    unittest.main()
