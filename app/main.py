import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from google.genai.errors import ClientError
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from app.core.config import Settings, get_settings
from app.services.pdf_ingestion import validate_pdf_upload
from app.services.rag import RagService
from app.stores.document_storage import create_document_storage
from app.stores.milvus import MilvusStore

logger = logging.getLogger(__name__)


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks: int


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_id: str | None = None
    document_ids: list[str] | None = None


class Citation(BaseModel):
    document_id: str
    filename: str
    page: int


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]


@lru_cache
def get_rag_service() -> RagService:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise HTTPException(503, "Set GEMINI_API_KEY before using ingestion or question endpoints")
    return RagService(settings, MilvusStore(settings))


app = FastAPI(title="PDF RAG API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=IngestResponse, status_code=201)
def upload_document(
    file: Annotated[UploadFile, File()],
    settings: Annotated[Settings, Depends(get_settings)],
    rag: Annotated[RagService, Depends(get_rag_service)],
    semantic_chunking: Annotated[bool | None, Form()] = None,
) -> IngestResponse:
    filename = Path(file.filename or "").name
    content = file.file.read(settings.max_upload_bytes + 1)
    try:
        pages = validate_pdf_upload(filename, content, settings.max_upload_bytes)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    if not any(text for _, text in pages):
        raise HTTPException(422, "PDF contains no extractable text")

    document_id = uuid4().hex
    try:
        storage = create_document_storage(settings)
        storage.save(document_id, content)
        count = rag.ingest(
            document_id,
            filename,
            pages,
            semantic_chunking=semantic_chunking,
        )
    except Exception as error:
        if isinstance(error, ClientError) and error.code == 429:
            logger.warning("Gemini quota exhausted during document ingestion")
            raise HTTPException(
                429,
                "Gemini embedding quota exhausted; wait for quota reset or review rate limits "
                "and billing",
            ) from error
        logger.exception("Document ingestion failed")
        raise HTTPException(
            503, "Document ingestion failed; check Gemini and Milvus connectivity"
        ) from error
    return IngestResponse(document_id=document_id, filename=filename, chunks=count)


@app.post("/questions", response_model=AnswerResponse)
def ask_question(
    request: QuestionRequest,
    rag: Annotated[RagService, Depends(get_rag_service)],
) -> AnswerResponse:
    try:
        return AnswerResponse(
            **rag.answer(
                request.question,
                document_id=request.document_id,
                document_ids=request.document_ids,
            )
        )
    except Exception as error:
        if isinstance(error, ClientError) and error.code == 429:
            logger.warning("Gemini quota exhausted during question answering")
            raise HTTPException(
                429,
                "Gemini API quota exhausted; wait for quota reset or review rate limits and billing",
            ) from error
        logger.exception("Question answering failed")
        raise HTTPException(
            503, "Question answering failed; check Gemini and Milvus connectivity"
        ) from error


@app.post("/questions/stream")
def ask_question_stream(
    request: QuestionRequest,
    rag: Annotated[RagService, Depends(get_rag_service)],
) -> StreamingResponse:
    try:
        answer_events = rag.stream_answer(
            request.question,
            document_id=request.document_id,
            document_ids=request.document_ids,
        )
    except Exception as error:
        if isinstance(error, ClientError) and error.code == 429:
            logger.warning("Gemini quota exhausted during streaming question answering")
            raise HTTPException(
                429,
                "Gemini API quota exhausted; wait for quota reset or review rate limits and billing",
            ) from error
        logger.exception("Streaming question answering failed before stream started")
        raise HTTPException(
            503, "Question answering failed; check Gemini and Milvus connectivity"
        ) from error

    def encode_events():
        try:
            for event_name, payload in answer_events:
                yield f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        except ClientError as error:
            if error.code == 429:
                logger.warning("Gemini quota exhausted during streaming question answering")
                payload = {"detail": "Gemini API quota exhausted; wait and retry"}
            else:
                logger.exception("Streaming question answering failed")
                payload = {"detail": "Question answering failed"}
            yield f"event: error\ndata: {json.dumps(payload)}\n\n"
        except Exception:
            logger.exception("Streaming question answering failed")
            yield 'event: error\ndata: {"detail":"Question answering failed"}\n\n'

    return StreamingResponse(
        encode_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
