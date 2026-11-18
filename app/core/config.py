from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_environment: Literal["development", "production"] = "development"
    cognito_auth_enabled: bool = False
    cognito_region: str = ""
    cognito_user_pool_id: str = ""
    cognito_app_client_id: str = ""
    cors_allowed_origins: list[str] = Field(default_factory=list)
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

    @model_validator(mode="after")
    def validate_deployment_configuration(self) -> "Settings":
        if "*" in self.cors_allowed_origins:
            raise ValueError("CORS_ALLOWED_ORIGINS must list explicit origins, not '*'.")

        if self.app_environment == "production":
            if self.document_storage_backend == "r2" and not all(
                (
                    self.r2_account_id,
                    self.r2_bucket,
                    self.r2_access_key_id,
                    self.r2_secret_access_key,
                )
            ):
                raise ValueError(
                    "R2 storage requires account ID, bucket, and access credentials."
                )

            missing = [
                name
                for name, value in (
                    ("COGNITO_AUTH_ENABLED", self.cognito_auth_enabled),
                    ("COGNITO_REGION", self.cognito_region),
                    ("COGNITO_USER_POOL_ID", self.cognito_user_pool_id),
                    ("COGNITO_APP_CLIENT_ID", self.cognito_app_client_id),
                    ("GEMINI_API_KEY", self.gemini_api_key),
                )
                if not value
            ]
            milvus_host = urlparse(self.milvus_uri).hostname
            if not milvus_host or milvus_host.lower() in {
                "localhost",
                "127.0.0.1",
                "::1",
            }:
                missing.append("a non-local MILVUS_URI")
            if missing:
                raise ValueError(
                    "Production configuration is incomplete: " + ", ".join(missing)
                )

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
