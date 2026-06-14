from __future__ import annotations

import unittest

from policychain.ingestion.loaders import PolicyLoadError, read_policy_file
from tests.helpers import SAMPLE_MANIFEST_PATH, SAMPLE_PDF_PATH


class LoaderTests(unittest.TestCase):
    def test_read_real_pdf_extracts_text(self) -> None:
        loaded = read_policy_file(SAMPLE_PDF_PATH)

        self.assertEqual(loaded.file_type, "pdf")
        self.assertIsNotNone(loaded.page_count)
        self.assertGreater(loaded.page_count or 0, 0)
        self.assertGreater(len(loaded.text), 500)
        self.assertIn("生成式人工智能", loaded.text)

    def test_unsupported_file_type_has_clear_error(self) -> None:
        with self.assertRaisesRegex(PolicyLoadError, "Unsupported policy file type"):
            read_policy_file(SAMPLE_MANIFEST_PATH)


if __name__ == "__main__":
    unittest.main()
