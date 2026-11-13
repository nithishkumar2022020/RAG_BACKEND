from unittest.mock import patch

from app.core.config import Settings
from app.services.rag import SYSTEM_PROMPT, RagService


class FakeStore:
    def search(self, vector, limit, document_id=None, document_ids=None):
        return [{"id": "chunk-1", "document_id": "doc-1", "filename": "notes.pdf",
                 "page": 2, "chunk_index": 0, "text": "Useful excerpt."}]

    def all_chunks(self, document_id=None, document_ids=None):
        return [{"id": "chunk-1", "document_id": "doc-1", "filename": "notes.pdf",
                 "page": 2, "chunk_index": 0, "text": "Useful excerpt."}]


def test_rag_streams_gemini_text_then_citations():
    settings = Settings(
        gemini_api_key="test-key",
        chunking_strategy="fixed",
        embedding_requests_per_minute=99,
        _env_file=None,
    )
    with patch("app.services.rag.genai.Client") as client_factory:
        service = RagService(settings, FakeStore())
        service._embed = lambda texts, task: [[1.0, 0.0] for _ in texts]
        chunks = [
            type("Response", (), {"text": "First "})(),
            type("Response", (), {"text": "second."})(),
        ]
        service.client.models.generate_content_stream.return_value = iter(chunks)

        events = list(service.stream_answer("Summarize this document"))

    call = client_factory.return_value.models.generate_content_stream.call_args
    assert call.kwargs["config"].system_instruction == SYSTEM_PROMPT
    assert call.kwargs["contents"].startswith("Question: Summarize this document")
    assert events == [
        ("answer_chunk", {"text": "First "}),
        ("answer_chunk", {"text": "second."}),
        (
            "citations",
            {
                "citations": [
                    {"document_id": "doc-1", "filename": "notes.pdf", "page": 2}
                ]
            },
        ),
    ]
