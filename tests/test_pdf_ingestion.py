import unittest
from unittest.mock import patch

from google.genai.errors import ClientError

from app.core.config import Settings
from app.services.pdf_ingestion import (
    build_chunks,
    chunk_text,
    extract_pdf,
    semantic_chunk_sentences,
    split_sentences,
    validate_pdf_upload,
)
from app.services.rag import RagService


class ChunkTextTests(unittest.TestCase):
    def test_chunks_have_configured_size_and_overlap(self):
        chunks = chunk_text("x" * 2500, size=1000, overlap=200)

        self.assertEqual([len(chunk) for chunk in chunks], [1000, 1000, 900])
        self.assertEqual(chunks[0][-200:], chunks[1][:200])

    def test_short_and_blank_text(self):
        self.assertEqual(chunk_text("  short  "), ["short"])
        self.assertEqual(chunk_text(" \n "), [])

    def test_rejects_invalid_size_or_overlap(self):
        for size, overlap in ((0, 0), (10, -1), (10, 10)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                chunk_text("text", size, overlap)

    def test_chunks_preserve_page_and_document_metadata(self):
        chunks = build_chunks("doc", "notes.pdf", [(3, "first page"), (4, "second page")])

        self.assertEqual([(item["page"], item["chunk_index"]) for item in chunks], [(3, 0), (4, 0)])
        self.assertTrue(all(item["document_id"] == "doc" for item in chunks))

    def test_splits_sentences_and_keeps_paragraphs_as_units(self):
        self.assertEqual(
            split_sentences("First sentence.\nContinued line.\n\nNew paragraph!"),
            ["First sentence.", "Continued line.", "New paragraph!"],
        )

    def test_semantic_chunking_splits_at_embedding_distance(self):
        chunks = semantic_chunk_sentences(
            ["Cats are pets.", "Dogs are pets.", "Stocks fell today."],
            [[1, 0], [0.99, 0.01], [0, 1]],
            size=100,
            overlap=0,
            breakpoint_percentile=50,
        )

        self.assertEqual(chunks, ["Cats are pets. Dogs are pets.", "Stocks fell today."])

    def test_semantic_chunking_obeys_size_and_overlap(self):
        chunks = semantic_chunk_sentences(
            ["alpha beta", "gamma delta", "epsilon zeta"],
            [[1, 0], [1, 0], [1, 0]],
            size=16,
            overlap=4,
            breakpoint_percentile=90,
        )

        self.assertTrue(all(len(chunk) <= 16 for chunk in chunks))
        self.assertEqual(chunks[0], "alpha beta")
        self.assertTrue(chunks[1].startswith("beta"))

    def test_semantic_chunking_rejects_mismatched_embeddings(self):
        with self.assertRaisesRegex(ValueError, "exactly one embedding"):
            semantic_chunk_sentences(["one", "two"], [[1, 0]])

    def test_semantic_chunking_does_not_insert_spaces_inside_long_sentences(self):
        chunks = semantic_chunk_sentences(
            ["abcdefghijklmnop"],
            [[1, 0]],
            size=8,
            overlap=0,
        )

        self.assertEqual(chunks, ["abcdefgh", "ijklmnop"])

    def test_upload_validation_rejects_wrong_extension_and_oversize(self):
        with self.assertRaisesRegex(ValueError, "Only PDF"):
            validate_pdf_upload("notes.txt", b"%PDF-", 10)
        with self.assertRaisesRegex(ValueError, "upload limit"):
            validate_pdf_upload("notes.pdf", b"%PDF-oversize", 5)

    def test_extracts_text_with_one_based_page_numbers(self):
        import pymupdf as fitz

        document = fitz.open()
        document.new_page().insert_text((72, 72), "first page")
        document.new_page().insert_text((72, 72), "second page")
        content = document.tobytes()
        document.close()

        self.assertEqual(extract_pdf(content), [(1, "first page"), (2, "second page")])


class EmbeddingBatchTests(unittest.TestCase):
    def test_ingestion_stores_authenticated_owner_on_each_chunk(self):
        settings = Settings(
            gemini_api_key="test-key",
            chunking_strategy="fixed",
            _env_file=None,
        )
        captured = {}

        class FakeStore:
            def insert(self, chunks, vectors):
                captured["chunks"] = chunks

        with patch("app.services.rag.genai.Client"):
            service = RagService(settings, FakeStore())
        with patch.object(service, "_embed_batches", return_value=[[1.0, 0.0]]):
            service.ingest(
                "doc-1",
                "notes.pdf",
                [(1, "A searchable sentence.")],
                owner_id="user-1",
            )

        assert captured["chunks"][0]["owner_id"] == "user-1"

    def test_ingestion_can_disable_semantic_chunking_per_document(self):
        settings = Settings(
            gemini_api_key="test-key",
            chunking_strategy="semantic",
            _env_file=None,
        )
        store = type("FakeStore", (), {"insert": lambda self, chunks, vectors: None})()

        with patch("app.services.rag.genai.Client"):
            service = RagService(settings, store)
        with (
            patch.object(service, "_semantic_chunk_text") as semantic_chunk_text,
            patch.object(service, "_embed_batches", return_value=[[1.0, 0.0]]) as embed_batches,
        ):
            count = service.ingest(
                "doc-1",
                "notes.pdf",
                [(1, "A first sentence. A second sentence.")],
                semantic_chunking=False,
            )

        self.assertEqual(count, 1)
        semantic_chunk_text.assert_not_called()
        self.assertEqual(
            embed_batches.call_args.args[0],
            ["A first sentence. A second sentence."],
        )

    def test_ingestion_can_enable_semantic_chunking_per_document(self):
        settings = Settings(
            gemini_api_key="test-key",
            chunking_strategy="fixed",
            _env_file=None,
        )
        store = type("FakeStore", (), {"insert": lambda self, chunks, vectors: None})()

        with patch("app.services.rag.genai.Client"):
            service = RagService(settings, store)
        with (
            patch.object(
                service,
                "_semantic_chunk_text",
                return_value=["semantic chunk"],
            ) as semantic_chunk_text,
            patch.object(service, "_embed_batches", return_value=[[1.0, 0.0]]),
        ):
            count = service.ingest(
                "doc-1",
                "notes.pdf",
                [(1, "Some source text.")],
                semantic_chunking=True,
            )

        self.assertEqual(count, 1)
        semantic_chunk_text.assert_called_once_with("Some source text.")

    def test_sentences_are_batched_at_configured_max_size(self):
        settings = Settings(
            gemini_api_key="test-key",
            embedding_batch_size=100,
            embedding_requests_per_minute=99,
            _env_file=None,
        )

        with patch("app.services.rag.genai.Client") as client_factory:
            service = RagService(settings, store=None)
            calls = []

            def embed_content(*, model, contents, config):
                calls.append(len(contents))
                return type(
                    "EmbeddingResponse",
                    (),
                    {
                        "embeddings": [
                            type("Embedding", (), {"values": [1.0, 0.0]})()
                            for _ in contents
                        ]
                    },
                )()

            service.client.models.embed_content.side_effect = embed_content
            with patch("app.services.rag.time.sleep"):
                embeddings = service._embed_batches(
                    [f"sentence {index}" for index in range(201)],
                    "RETRIEVAL_DOCUMENT",
                )

        self.assertEqual(calls, [100, 100, 1])
        self.assertEqual(len(embeddings), 201)
        client_factory.assert_called_once()
        client_kwargs = client_factory.call_args.kwargs
        self.assertEqual(client_kwargs["api_key"], "test-key")
        self.assertEqual(client_kwargs["http_options"].api_version, "v1beta")

    def test_embedding_batch_configuration_cannot_exceed_api_limit(self):
        with self.assertRaises(ValueError):
            Settings(embedding_batch_size=101, _env_file=None)

    def test_embedding_requests_are_spaced_by_configured_rate(self):
        settings = Settings(
            gemini_api_key="test-key",
            embedding_requests_per_minute=90,
            _env_file=None,
        )

        with patch("app.services.rag.genai.Client"):
            service = RagService(settings, store=None)
        with (
            patch("app.services.rag.time.monotonic", return_value=100.0),
            patch("app.services.rag.time.sleep") as sleep,
        ):
            service._wait_for_embedding_slot()
            service._wait_for_embedding_slot()

        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 60 / 90)

    def test_embedding_429_retries_after_provider_delay(self):
        settings = Settings(
            gemini_api_key="test-key",
            embedding_retry_attempts=1,
            _env_file=None,
        )
        error = ClientError(
            429,
            {
                "error": {
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": "2.5s",
                        }
                    ]
                }
            },
        )
        response = type(
            "EmbeddingResponse",
            (),
            {"embeddings": [type("Embedding", (), {"values": [1.0, 0.0]})()]},
        )()

        with patch("app.services.rag.genai.Client") as client_factory:
            service = RagService(settings, store=None)
            service.client.models.embed_content.side_effect = [error, response]
            with patch("app.services.rag.time.sleep") as sleep:
                embeddings = service._embed(["test"], "RETRIEVAL_QUERY")

        self.assertEqual(embeddings, [[1.0, 0.0]])
        self.assertEqual(client_factory.return_value.models.embed_content.call_count, 2)
        self.assertIn(2.5, [call.args[0] for call in sleep.call_args_list])


if __name__ == "__main__":
    unittest.main()
