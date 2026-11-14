import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from threading import BoundedSemaphore, Lock
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
INGESTION_WORKERS = 1
MAX_QUEUED_INGESTIONS = 1
ingestion_executor = ThreadPoolExecutor(
    max_workers=INGESTION_WORKERS,
    thread_name_prefix="document-ingestion",
)
ingestion_slots = BoundedSemaphore(INGESTION_WORKERS + MAX_QUEUED_INGESTIONS)
ingestion_jobs: dict[str, dict] = {}
ingestion_jobs_lock = Lock()


class IngestJobResponse(BaseModel):
    job_id: str
    document_id: str
    filename: str
    status: str
    chunks: int | None = None
    detail: str | None = None


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


def _update_ingestion_job(job_id: str, **updates: object) -> None:
    with ingestion_jobs_lock:
        ingestion_jobs[job_id].update(updates)


def _run_document_ingestion(
    job_id: str,
    document_id: str,
    filename: str,
    content: bytes,
    settings: Settings,
    rag: RagService,
    semantic_chunking: bool | None,
) -> None:
    _update_ingestion_job(job_id, status="processing")
    try:
        pages = validate_pdf_upload(filename, content, settings.max_upload_bytes)
        if not any(text for _, text in pages):
            raise ValueError("PDF contains no extractable text")

        storage = create_document_storage(settings)
        storage.save(document_id, content)
        count = rag.ingest(
            document_id,
            filename,
            pages,
            semantic_chunking=semantic_chunking,
        )
    except ValueError as error:
        logger.info("Document ingestion job %s rejected: %s", job_id, error)
        _update_ingestion_job(job_id, status="failed", detail=str(error))
    except ClientError as error:
        if error.code == 429:
            logger.warning("Gemini quota exhausted during document ingestion job %s", job_id)
            detail = (
                "Gemini embedding quota exhausted; wait for quota reset or review rate limits "
                "and billing"
            )
        else:
            logger.exception("Document ingestion job %s failed", job_id)
            detail = "Document ingestion failed; check Gemini and Milvus connectivity"
        _update_ingestion_job(job_id, status="failed", detail=detail)
    except Exception:
        logger.exception("Document ingestion job %s failed", job_id)
        _update_ingestion_job(
            job_id,
            status="failed",
            detail="Document ingestion failed; check Gemini and Milvus connectivity",
        )
    else:
        _update_ingestion_job(job_id, status="completed", chunks=count)
    finally:
        ingestion_slots.release()


@app.post("/documents", response_model=IngestJobResponse, status_code=202)
def upload_document(
    file: Annotated[UploadFile, File()],
    settings: Annotated[Settings, Depends(get_settings)],
    rag: Annotated[RagService, Depends(get_rag_service)],
    semantic_chunking: Annotated[bool | None, Form()] = None,
) -> IngestJobResponse:
    filename = Path(file.filename or "").name
    content = file.file.read(settings.max_upload_bytes + 1)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(422, "Only PDF files are supported")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            422,
            f"PDF exceeds the {settings.max_upload_bytes}-byte upload limit",
        )
    if not ingestion_slots.acquire(blocking=False):
        raise HTTPException(
            429,
            "Document ingestion queue is full; wait for a current upload to finish and retry",
            headers={"Retry-After": "30"},
        )

    job_id = uuid4().hex
    document_id = uuid4().hex
    with ingestion_jobs_lock:
        ingestion_jobs[job_id] = {
            "job_id": job_id,
            "document_id": document_id,
            "filename": filename,
            "status": "queued",
            "chunks": None,
            "detail": None,
        }

    try:
        ingestion_executor.submit(
            _run_document_ingestion,
            job_id,
            document_id,
            filename,
            content,
            settings,
            rag,
            semantic_chunking,
        )
    except RuntimeError as error:
        ingestion_slots.release()
        with ingestion_jobs_lock:
            ingestion_jobs.pop(job_id, None)
        logger.exception("Could not queue document ingestion job %s", job_id)
        raise HTTPException(503, "Document ingestion service is unavailable") from error

    return IngestJobResponse(
        job_id=job_id,
        document_id=document_id,
        filename=filename,
        status="queued",
    )


@app.get("/documents/{job_id}", response_model=IngestJobResponse)
def get_document_ingestion_status(job_id: str) -> IngestJobResponse:
    with ingestion_jobs_lock:
        job = ingestion_jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Document ingestion job not found")
        return IngestJobResponse(**job)


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
