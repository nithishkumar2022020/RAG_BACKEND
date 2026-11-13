from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pymupdf
from fastapi.testclient import TestClient
from google.genai.errors import ClientError

from app.core.config import Settings, get_settings
from app.main import app, get_rag_service
from app.stores.document_storage import R2DocumentStorage, create_document_storage


def test_question_response_has_stable_citation_shape():
    class FakeRagService:
        def answer(
            self,
            question: str,
            document_id: str | None = None,
            document_ids: list[str] | None = None,
        ) -> dict:
            return {
                "answer": "A grounded answer [notes.pdf, p. 2].",
                "citations": [{"document_id": "doc-1", "filename": "notes.pdf", "page": 2}],
            }

    app.dependency_overrides[get_rag_service] = FakeRagService
    try:
        response = TestClient(app).post("/questions", json={"question": "What happened?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["citations"] == [
        {"document_id": "doc-1", "filename": "notes.pdf", "page": 2}
    ]


def test_streaming_question_emits_answer_chunks_then_citations():
    class FakeRagService:
        def stream_answer(
            self,
            question: str,
            document_id: str | None = None,
            document_ids: list[str] | None = None,
        ):
            assert question == "What happened?"
            assert document_id == "doc-1"
            yield "answer_chunk", {"text": "A grounded "}
            yield "answer_chunk", {"text": "answer."}
            yield "citations", {
                "citations": [{"document_id": "doc-1", "filename": "notes.pdf", "page": 2}]
            }

    app.dependency_overrides[get_rag_service] = FakeRagService
    try:
        response = TestClient(app).post(
            "/questions/stream",
            json={"question": "What happened?", "document_id": "doc-1"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: answer_chunk\ndata: {\"text\": \"A grounded \"}" in response.text
    assert "event: answer_chunk\ndata: {\"text\": \"answer.\"}" in response.text
    assert 'event: citations\ndata: {"citations": [{"document_id": "doc-1"' in response.text


def test_question_failure_logs_error_but_returns_generic_503():
    class FailingRagService:
        def answer(
            self,
            question: str,
            document_id: str | None = None,
            document_ids: list[str] | None = None,
        ) -> dict:
            raise RuntimeError("internal provider details")

    app.dependency_overrides[get_rag_service] = FailingRagService
    try:
        with patch("app.main.logger.exception") as log_exception:
            response = TestClient(app).post(
                "/questions",
                json={"question": "What happened?"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Question answering failed; check Gemini and Milvus connectivity"
    )
    assert "internal provider details" not in response.text
    log_exception.assert_called_once_with("Question answering failed")


def test_question_quota_error_returns_429_without_provider_details():
    class QuotaLimitedRagService:
        def answer(
            self,
            question: str,
            document_id: str | None = None,
            document_ids: list[str] | None = None,
        ) -> dict:
            raise ClientError(429, {"error": {"message": "provider quota details"}})

    app.dependency_overrides[get_rag_service] = QuotaLimitedRagService
    try:
        with patch("app.main.logger.warning") as log_warning:
            response = TestClient(app).post(
                "/questions",
                json={"question": "What happened?"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 429
    assert "quota exhausted" in response.json()["detail"]
    assert "provider quota details" not in response.text
    log_warning.assert_called_once_with("Gemini quota exhausted during question answering")


def test_upload_quota_error_returns_429():
    class QuotaLimitedRagService:
        def ingest(self, document_id: str, filename: str, pages: list[tuple[int, str]]) -> int:
            raise ClientError(429, {"error": {"message": "provider quota details"}})

    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "searchable text")
    content = document.tobytes()
    document.close()
    with TemporaryDirectory() as directory:
        app.dependency_overrides[get_rag_service] = QuotaLimitedRagService
        app.dependency_overrides[get_settings] = lambda: Settings(
            data_dir=Path(directory),
            document_storage_backend="local",
            _env_file=None,
        )
        try:
            with patch("app.main.logger.warning") as log_warning:
                response = TestClient(app).post(
                    "/documents",
                    files={"file": ("notes.pdf", content, "application/pdf")},
                )
        finally:
            app.dependency_overrides.clear()

    assert response.status_code == 429
    assert "quota exhausted" in response.json()["detail"]
    assert "provider quota details" not in response.text
    log_warning.assert_called_once_with("Gemini quota exhausted during document ingestion")


def test_pdf_upload_extracts_and_stores_original():
    class FakeRagService:
        def ingest(self, document_id: str, filename: str, pages: list[tuple[int, str]]) -> int:
            assert filename == "notes.pdf"
            assert pages == [(1, "searchable text")]
            return 1

    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "searchable text")
    content = document.tobytes()
    document.close()

    with TemporaryDirectory() as directory:
        app.dependency_overrides[get_rag_service] = FakeRagService
        app.dependency_overrides[get_settings] = lambda: Settings(
            data_dir=Path(directory),
            document_storage_backend="local",
            _env_file=None,
        )
        try:
            response = TestClient(app).post(
                "/documents", files={"file": ("notes.pdf", content, "application/pdf")}
            )
        finally:
            app.dependency_overrides.clear()

        assert response.status_code == 201
        assert response.json()["chunks"] == 1
        assert len(list((Path(directory) / "documents").glob("*.pdf"))) == 1


def test_r2_storage_uploads_pdf_to_configured_bucket():
    class FakeS3Client:
        def __init__(self):
            self.upload = None

        def put_object(self, **kwargs):
            self.upload = kwargs

    storage = R2DocumentStorage.__new__(R2DocumentStorage)
    storage.bucket = "pdf-bucket"
    storage.prefix = "uploads"
    storage.client = FakeS3Client()

    key = storage.save("document-1", b"%PDF-content")

    assert key == "uploads/document-1.pdf"
    assert storage.client.upload == {
        "Bucket": "pdf-bucket",
        "Key": "uploads/document-1.pdf",
        "Body": b"%PDF-content",
        "ContentType": "application/pdf",
    }


def test_r2_storage_uses_cloudflare_s3_endpoint_and_credentials():
    with patch("boto3.client") as client_factory:
        R2DocumentStorage(
            "account-123",
            "pdf-bucket",
            "access-key",
            "secret-key",
            prefix="",
        )

    client_factory.assert_called_once()
    args, kwargs = client_factory.call_args
    assert args == ("s3",)
    assert kwargs["endpoint_url"] == "https://account-123.r2.cloudflarestorage.com"
    assert kwargs["region_name"] == "auto"
    assert kwargs["aws_access_key_id"] == "access-key"
    assert kwargs["aws_secret_access_key"] == "secret-key"
    assert kwargs["config"].signature_version == "s3v4"
    assert kwargs["config"].s3 == {"addressing_style": "path"}


def test_r2_storage_requires_credentials():
    settings = Settings(
        document_storage_backend="r2",
        r2_account_id="account",
        r2_bucket="bucket",
        r2_access_key_id="access",
        r2_secret_access_key="",
        _env_file=None,
    )

    with patch("app.stores.document_storage.R2DocumentStorage") as storage_class:
        try:
            create_document_storage(settings)
        except ValueError as error:
            assert "R2_SECRET_ACCESS_KEY" in str(error)
        else:
            raise AssertionError("Expected incomplete R2 settings to fail")
        storage_class.assert_not_called()
