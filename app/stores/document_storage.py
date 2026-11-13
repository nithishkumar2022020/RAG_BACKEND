from pathlib import Path
from typing import Protocol

from app.core.config import Settings


class DocumentStorage(Protocol):
    def save(self, document_id: str, content: bytes) -> str: ...


class LocalDocumentStorage:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def save(self, document_id: str, content: bytes) -> str:
        self.directory.mkdir(parents=True, exist_ok=True)
        key = f"{document_id}.pdf"
        (self.directory / key).write_bytes(content)
        return key


class R2DocumentStorage:
    def __init__(
        self,
        account_id: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        prefix: str = "documents",
    ) -> None:
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            region_name="auto",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )

    def save(self, document_id: str, content: bytes) -> str:
        filename = f"{document_id}.pdf"
        key = f"{self.prefix}/{filename}" if self.prefix else filename
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType="application/pdf",
        )
        return key


def create_document_storage(settings: Settings) -> DocumentStorage:
    if settings.document_storage_backend == "local":
        return LocalDocumentStorage(settings.data_dir / "documents")

    required = {
        "R2_ACCOUNT_ID": settings.r2_account_id,
        "R2_BUCKET": settings.r2_bucket,
        "R2_ACCESS_KEY_ID": settings.r2_access_key_id,
        "R2_SECRET_ACCESS_KEY": settings.r2_secret_access_key,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing R2 configuration: {', '.join(missing)}")
    return R2DocumentStorage(
        settings.r2_account_id,
        settings.r2_bucket,
        settings.r2_access_key_id,
        settings.r2_secret_access_key,
        settings.r2_prefix,
    )
