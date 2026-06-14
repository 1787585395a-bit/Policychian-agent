from __future__ import annotations

import unittest

from policychain.ingestion.id_generator import compute_file_hash
from policychain.ingestion.loaders import read_policy_file
from policychain.ingestion.metadata_extractor import extract_metadata, find_manifest_record
from tests.helpers import SAMPLE_MANIFEST_PATH, SAMPLE_PDF_PATH


class MetadataExtractorTests(unittest.TestCase):
    def test_find_manifest_record_by_filename(self) -> None:
        record = find_manifest_record(
            SAMPLE_MANIFEST_PATH,
            "48_国家_2023_生成式人工智能服务管理暂行办法.pdf",
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.sequence, 48)
        self.assertEqual(record.title, "生成式人工智能服务管理暂行办法")
        self.assertEqual(record.document_number, "第15号")

    def test_extract_metadata_from_manifest(self) -> None:
        loaded = read_policy_file(SAMPLE_PDF_PATH)
        metadata = extract_metadata(
            loaded_file=loaded,
            file_hash=compute_file_hash(SAMPLE_PDF_PATH),
            manifest_path=SAMPLE_MANIFEST_PATH,
        )

        self.assertEqual(metadata.policy_id, "POL-2023-NAT-0048")
        self.assertEqual(metadata.title, "生成式人工智能服务管理暂行办法")
        self.assertEqual(metadata.document_number, "第15号")
        self.assertEqual(metadata.publish_date, "2023-05-23")
        self.assertEqual(metadata.issuing_agencies, ["国家级政府或主管部门（按官方来源）"])
        self.assertEqual(metadata.source_url, "https://www.gov.cn/zhengce/zhengceku/202307/content_6891752.htm")
        self.assertEqual(metadata.policy_level, "national")
        self.assertEqual(metadata.original_filename, SAMPLE_PDF_PATH.name)
        self.assertTrue(metadata.file_hash)


if __name__ == "__main__":
    unittest.main()
