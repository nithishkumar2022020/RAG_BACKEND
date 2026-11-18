from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from app.core.auth import require_authenticated_client
from app.core.config import Settings, get_settings
from app.main import app, get_rag_service, ingestion_jobs, ingestion_jobs_lock


@pytest.fixture
def cognito_test_keys(monkeypatch):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    class FakeJwksClient:
        def get_signing_key_from_jwt(self, token: str):
            return SimpleNamespace(key=public_key)

    monkeypatch.setattr("app.core.auth._jwks_client", lambda issuer: FakeJwksClient())
    return private_key


def _settings(**overrides) -> Settings:
    return Settings(
        cognito_auth_enabled=True,
        cognito_region="ap-southeast-2",
        cognito_user_pool_id="ap-southeast-2_testpool",
        cognito_app_client_id="test-client",
        _env_file=None,
        **overrides,
    )


def _token(private_key, **overrides) -> str:
    now = datetime.now(UTC)
    claims = {
        "iss": "https://cognito-idp.ap-southeast-2.amazonaws.com/ap-southeast-2_testpool",
        "sub": "user-123",
        "token_use": "access",
        "client_id": "test-client",
        "iat": now,
        "exp": now + timedelta(minutes=5),
        **overrides,
    }
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-key"})


def _credentials(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_authentication_is_skipped_when_disabled_for_local_development():
    settings = Settings(_env_file=None)

    assert require_authenticated_client(None, settings) is None


def test_cognito_access_token_is_accepted(cognito_test_keys):
    claims = require_authenticated_client(
        _credentials(_token(cognito_test_keys)),
        _settings(),
    )

    assert claims is not None
    assert claims["sub"] == "user-123"


def test_invalid_cognito_signature_is_rejected(cognito_test_keys):
    other_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    with pytest.raises(HTTPException) as error:
        require_authenticated_client(
            _credentials(_token(other_private_key)),
            _settings(),
        )

    assert error.value.status_code == 401


def test_wrong_cognito_issuer_is_rejected(cognito_test_keys):
    with pytest.raises(HTTPException) as error:
        require_authenticated_client(
            _credentials(_token(cognito_test_keys, iss="https://wrong-issuer.example")),
            _settings(),
        )

    assert error.value.status_code == 401


@pytest.mark.parametrize(
    "claim_overrides",
    [
        {"token_use": "id"},
        {"client_id": "different-client"},
        {"sub": ""},
    ],
)
def test_invalid_identity_claims_are_rejected(cognito_test_keys, claim_overrides):
    with pytest.raises(HTTPException) as error:
        require_authenticated_client(
            _credentials(_token(cognito_test_keys, **claim_overrides)),
            _settings(),
        )

    assert error.value.status_code == 401


def test_expired_cognito_token_is_rejected(cognito_test_keys):
    expired = datetime.now(UTC) - timedelta(minutes=1)

    with pytest.raises(HTTPException) as error:
        require_authenticated_client(
            _credentials(_token(cognito_test_keys, exp=expired)),
            _settings(),
        )

    assert error.value.status_code == 401


def test_missing_bearer_token_is_rejected_when_authentication_enabled():
    with pytest.raises(HTTPException) as error:
        require_authenticated_client(None, _settings())

    assert error.value.status_code == 401
    assert error.value.headers == {"WWW-Authenticate": "Bearer"}


def test_question_endpoint_requires_token_when_cognito_auth_is_enabled():
    app.dependency_overrides[get_settings] = lambda: _settings()
    try:
        response = TestClient(app).post("/questions", json={"question": "hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_ingestion_job_status_is_hidden_from_other_authenticated_users():
    job_id = "auth-isolation-test"
    with ingestion_jobs_lock:
        ingestion_jobs[job_id] = {
            "job_id": job_id,
            "document_id": "doc-1",
            "filename": "private.pdf",
            "status": "completed",
            "chunks": 1,
            "detail": None,
            "owner_id": "owner-1",
        }
    app.dependency_overrides[require_authenticated_client] = lambda: {"sub": "owner-2"}
    try:
        response = TestClient(app).get(f"/documents/{job_id}")
    finally:
        app.dependency_overrides.clear()
        with ingestion_jobs_lock:
            ingestion_jobs.pop(job_id, None)

    assert response.status_code == 404


def test_retrieval_is_scoped_to_verified_cognito_sub():
    rag = Mock()
    rag.answer.return_value = {"answer": "No relevant PDF content was found.", "citations": []}
    app.dependency_overrides[require_authenticated_client] = lambda: {"sub": "owner-2"}
    app.dependency_overrides[get_rag_service] = lambda: rag
    try:
        response = TestClient(app).post(
            "/questions",
            json={"question": "Find this document", "document_id": "owner-1-document"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert rag.answer.call_args.kwargs["owner_id"] == "owner-2"
    assert rag.answer.call_args.kwargs["document_id"] == "owner-1-document"


def test_production_settings_fail_closed_without_required_integrations():
    with pytest.raises(ValueError, match="Production configuration is incomplete"):
        Settings(app_environment="production", _env_file=None)


def test_production_settings_reject_disabled_cognito_auth():
    with pytest.raises(ValueError, match="COGNITO_AUTH_ENABLED"):
        Settings(
            app_environment="production",
            cognito_auth_enabled=False,
            cognito_region="ap-southeast-2",
            cognito_user_pool_id="ap-southeast-2_pool",
            cognito_app_client_id="client",
            gemini_api_key="configured",
            milvus_uri="https://cluster.example.test",
            _env_file=None,
        )


def test_production_settings_accept_complete_remote_configuration():
    settings = Settings(
        app_environment="production",
        cognito_auth_enabled=True,
        cognito_region="ap-southeast-2",
        cognito_user_pool_id="ap-southeast-2_pool",
        cognito_app_client_id="client",
        gemini_api_key="configured",
        milvus_uri="https://cluster.example.test",
        _env_file=None,
    )

    assert settings.app_environment == "production"


def test_wildcard_cors_origin_is_rejected():
    with pytest.raises(ValueError, match="explicit origins"):
        Settings(cors_allowed_origins=["*"], _env_file=None)


def test_r2_storage_requires_all_credentials():
    with pytest.raises(ValueError, match="R2 storage requires"):
        Settings(
            app_environment="production",
            cognito_auth_enabled=True,
            cognito_region="ap-southeast-2",
            cognito_user_pool_id="pool",
            cognito_app_client_id="client",
            gemini_api_key="configured",
            milvus_uri="https://cluster.example.test",
            document_storage_backend="r2",
            r2_account_id="",
            r2_bucket="",
            r2_access_key_id="",
            r2_secret_access_key="",
            _env_file=None,
        )
