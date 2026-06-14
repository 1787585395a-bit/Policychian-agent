from __future__ import annotations

import unittest

from policychain.ingestion.chunker import chunk_policy_text


class ChunkerTests(unittest.TestCase):
    def test_chunk_policy_text_generates_stable_ids_and_neighbors(self) -> None:
        text = """
        第一章 总则
        第一条 为了规范生成式人工智能服务，促进生成式人工智能健康发展，制定本办法。
        第二条 在中华人民共和国境内提供生成式人工智能服务，适用本办法。

        第二章 技术发展与治理
        第三条 国家坚持发展和安全并重，促进创新和依法治理。
        第四条 提供者应当依法承担网络信息内容安全责任。
        """

        chunks = chunk_policy_text("POL-2023-NAT-0048", text, max_chars=80)

        self.assertTrue(chunks)
        self.assertEqual(chunks[0].chunk_id, "POL-2023-NAT-0048-S01-C001")
        self.assertIsNone(chunks[0].previous_chunk_id)
        self.assertEqual(chunks[0].next_chunk_id, chunks[1].chunk_id)
        self.assertIsNone(chunks[-1].next_chunk_id)
        self.assertTrue(all(chunk.char_start < chunk.char_end for chunk in chunks))
        self.assertTrue(any("规范生成式人工智能服务" in chunk.content for chunk in chunks))

    def test_chunk_policy_text_keeps_section_titles(self) -> None:
        chunks = chunk_policy_text(
            "POL-2023-NAT-0048",
            "第一章 总则\n第一条 内容一。\n第二章 附则\n第二条 内容二。",
            max_chars=50,
        )

        titles = {chunk.section_title for chunk in chunks}
        self.assertIn("第一章 总则", titles)
        self.assertIn("第二章 附则", titles)


if __name__ == "__main__":
    unittest.main()
