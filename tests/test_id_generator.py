from __future__ import annotations

import unittest

from policychain.ingestion.id_generator import (
    compute_file_hash,
    generate_chunk_id,
    generate_policy_id,
)
from tests.helpers import SAMPLE_PDF_PATH


class IDGeneratorTests(unittest.TestCase):
    def test_compute_file_hash_is_stable(self) -> None:
        first = compute_file_hash(SAMPLE_PDF_PATH)
        second = compute_file_hash(SAMPLE_PDF_PATH)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_generate_policy_id_without_region(self) -> None:
        self.assertEqual(generate_policy_id(2023, "NAT", 48), "POL-2023-NAT-0048")

    def test_generate_policy_id_with_region(self) -> None:
        self.assertEqual(generate_policy_id("2025", "MIIT", "2", region_code="GD"), "POL-2025-GD-MIIT-0002")

    def test_generate_chunk_id_zero_pads_section_and_chunk(self) -> None:
        chunk_id = generate_chunk_id("POL-2023-NAT-0048", section_index=3, chunk_index=5)

        self.assertEqual(chunk_id, "POL-2023-NAT-0048-S03-C005")


if __name__ == "__main__":
    unittest.main()
