from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_generation_model: str = "gemini-2.5-flash"
    embedding_batch_size: int = Field(default=100, gt=0, le=100)
    embedding_requests_per_minute: int = Field(default=30, gt=0, le=99)
    embedding_retry_attempts: int = Field(default=3, ge=0, le=5)
    embedding_dimension: int = Field(default=768, gt=0)
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_collection: str = "rag_chunks_v2"
    data_dir: Path = Path("data")
    document_storage_backend: Literal["local", "r2"] = "local"
    r2_account_id: str = ""
    r2_bucket: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_prefix: str = "documents"
    chunking_strategy: Literal["semantic", "fixed"] = "semantic"
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    semantic_breakpoint_percentile: int = Field(default=90, ge=0, le=100)
    retrieval_candidates: int = Field(default=20, gt=0)
    top_k: int = Field(default=5, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
