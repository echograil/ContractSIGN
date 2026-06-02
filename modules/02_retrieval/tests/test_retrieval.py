from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from retrieval_module import RetrievalModule, TextChunk, load_chunks_from_directory, retrieve_directory


class RetrievalModuleTests(unittest.TestCase):
    def test_empty_index_returns_empty(self) -> None:
        module = RetrievalModule()

        self.assertEqual(module.retrieve("governing law"), [])

    def test_empty_index_call_is_no_op(self) -> None:
        module = RetrievalModule()

        module.index([])

        self.assertEqual(module.retrieve("governing law"), [])

    def test_scores_are_normalized_and_sorted(self) -> None:
        module = RetrievalModule()
        module.index(
            [
                make_chunk(
                    "This Agreement shall be governed by the laws of Delaware.",
                    chunk_index=0,
                ),
                make_chunk(
                    "The supplier shall maintain insurance coverage.",
                    chunk_index=1,
                ),
            ]
        )

        results = module.retrieve("Which law governs this contract?", top_k=2)
        scores = [result.score for result in results]

        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(results[0].strategy_tag, "hybrid_v0")

    def test_top_k_does_not_exceed_index_size(self) -> None:
        module = RetrievalModule()
        module.index([make_chunk("Agreement law.", 0), make_chunk("Insurance policy.", 1)])

        results = module.retrieve("agreement", top_k=10)

        self.assertLessEqual(len(results), 2)

    def test_fields_are_passed_through(self) -> None:
        original = make_chunk(
            "Neither party may assign this Agreement without prior written consent.",
            chunk_index=7,
            source_file="fileA.pdf",
            location="p.9",
            clause_id=None,
        )
        module = RetrievalModule()
        module.index([original])

        result = module.retrieve("assignment consent", top_k=1)[0]

        self.assertIs(result.chunk, original)
        self.assertEqual(result.chunk.source_file, "fileA.pdf")
        self.assertEqual(result.chunk.location, "p.9")
        self.assertIsNone(result.chunk.clause_id)

    def test_long_query_does_not_crash(self) -> None:
        module = RetrievalModule()
        module.index([make_chunk("This contract includes an insurance requirement.", 0)])

        with self.assertLogs("retrieval_module.retrieval", level="WARNING"):
            results = module.retrieve("contract " * 2000, top_k=3)

        self.assertIsInstance(results, list)

    def test_duplicate_index_results_are_deduplicated(self) -> None:
        module = RetrievalModule()
        chunk = make_chunk("The agreement requires audit rights.", 0, source_file="a.pdf")
        module.index([chunk])
        module.index([chunk])

        results = module.retrieve("audit rights", top_k=5)
        keys = [(result.chunk.source_file, result.chunk.chunk_index) for result in results]

        self.assertEqual(len(keys), len(set(keys)))

    def test_filter_doc_type(self) -> None:
        module = RetrievalModule()
        module.index(
            [
                make_chunk("Contract contains audit rights.", 0, doc_type="contract"),
                make_chunk("FAQ contains audit rights.", 1, doc_type="faq"),
            ]
        )

        results = module.retrieve("audit rights", filter_doc_type="contract")

        self.assertTrue(results)
        self.assertTrue(all(result.chunk.doc_type == "contract" for result in results))

    def test_filter_source_file(self) -> None:
        module = RetrievalModule()
        module.index(
            [
                make_chunk("Contract contains audit rights.", 0, source_file="fileA.pdf"),
                make_chunk("Contract contains audit rights.", 1, source_file="fileB.pdf"),
            ]
        )

        results = module.retrieve("audit rights", filter_source_file="fileA.pdf")

        self.assertTrue(results)
        self.assertTrue(all(result.chunk.source_file == "fileA.pdf" for result in results))

    def test_keyword_only_legal_term_smoke(self) -> None:
        module = RetrievalModule()
        module.index(
            [
                make_chunk(
                    "During the term, neither party shall non-solicit employees of the other party.",
                    0,
                ),
                make_chunk("This page discusses governing law only.", 1),
            ]
        )

        results = module.retrieve("non-solicit employees", top_k=3)

        self.assertTrue(any("solicit" in result.chunk.text.lower() for result in results))

    def test_chinese_legal_queries_expand_to_english_contract_terms(self) -> None:
        module = RetrievalModule()
        module.index(
            [
                make_chunk("Party B shall pay the service fee within 30 days.", 0),
                make_chunk("Either party may terminate this Agreement upon written notice.", 1),
                make_chunk("Neither party may assign this Agreement without prior written consent.", 2),
                make_chunk("This Agreement describes the project scope and cooperation obligations.", 3),
            ]
        )

        cases = [
            ("付款义务是什么", "service fee"),
            ("合同什么时候可以终止", "terminate"),
            ("能不能转让合同", "assign"),
            ("这份合同主要内容是什么", "project scope"),
        ]
        for query, expected in cases:
            with self.subTest(query=query):
                results = module.retrieve(query, top_k=2)
                rendered = " ".join(result.chunk.text.lower() for result in results)
                self.assertIn(expected, rendered)

    def test_chinese_query_without_known_terms_still_returns_contract_context(self) -> None:
        module = RetrievalModule()
        module.index(
            [
                make_chunk("This Agreement is between Party A and Party B.", 0),
                make_chunk("The supplier shall maintain insurance coverage.", 1),
            ]
        )

        results = module.retrieve("这是什么", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertIn("Agreement", results[0].chunk.text)

    def test_invalid_doc_type_is_passed_through_with_warning(self) -> None:
        module = RetrievalModule()

        with self.assertLogs("retrieval_module.retrieval", level="WARNING"):
            module.index([make_chunk("Some text", 0, doc_type="custom")])

        results = module.retrieve("text", top_k=1)
        self.assertEqual(results[0].chunk.doc_type, "custom")

    def test_retrieve_directory_writes_txt_answer_and_manifest(self) -> None:
        root_dir = Path(tempfile.mkdtemp(prefix="contractsign-retrieval-"))
        self.addCleanup(lambda: shutil.rmtree(root_dir))
        input_dir = root_dir / "input"
        output_dir = root_dir / "output"
        input_dir.mkdir()
        chunks = [
            make_chunk("This Agreement shall be governed by Delaware law.", 0),
            make_chunk("The supplier shall maintain insurance.", 1),
        ]
        (input_dir / "chunks.json").write_text(
            json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        (input_dir / "question.txt").write_text(
            "Which law governs this contract?",
            encoding="utf-8",
        )

        result = retrieve_directory(input_dir=input_dir, output_dir=output_dir, top_k=1)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.chunk_count, 2)
        self.assertEqual(result.result_count, 1)
        answer = (output_dir / "answer.txt").read_text(encoding="utf-8")
        self.assertIn("问题：", answer)
        self.assertIn("Delaware law", answer)
        self.assertTrue((output_dir / "retrieval_results.json").exists())
        self.assertTrue((output_dir / "manifest.json").exists())

    def test_load_chunks_from_directory_falls_back_to_per_file_chunks(self) -> None:
        root_dir = Path(tempfile.mkdtemp(prefix="contractsign-retrieval-"))
        self.addCleanup(lambda: shutil.rmtree(root_dir))
        input_dir = root_dir / "input"
        input_dir.mkdir()
        (input_dir / "a.chunks.json").write_text(
            json.dumps([asdict(make_chunk("Audit rights.", 0, source_file="a.pdf"))]),
            encoding="utf-8",
        )
        (input_dir / "b.chunks.json").write_text(
            json.dumps([asdict(make_chunk("Insurance coverage.", 0, source_file="b.pdf"))]),
            encoding="utf-8",
        )

        chunks = load_chunks_from_directory(input_dir)

        self.assertEqual([chunk.source_file for chunk in chunks], ["a.pdf", "b.pdf"])


def make_chunk(
    text: str,
    chunk_index: int,
    source_file: str = "contract.pdf",
    location: str = "p.1",
    doc_type: str = "contract",
    clause_id: str | None = None,
) -> TextChunk:
    return TextChunk(
        text=text,
        source_file=source_file,
        location=location,
        doc_type=doc_type,
        chunk_index=chunk_index,
        clause_id=clause_id,
    )


if __name__ == "__main__":
    unittest.main()
