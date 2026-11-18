from functools import lru_cache
from typing import Annotated, Any

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient, PyJWKClientConnectionError, PyJWKClientError
from jwt.exceptions import InvalidTokenError

from app.core.config import Settings, get_settings

bearer_scheme = HTTPBearer(auto_error=False)


@lru_cache(maxsize=4)
def _jwks_client(issuer: str) -> PyJWKClient:
    return PyJWKClient(
        f"{issuer}/.well-known/jwks.json",
        cache_jwk_set=True,
        timeout=5,
    )


def require_authenticated_client(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any] | None:
    if not settings.cognito_auth_enabled:
        return None
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            401,
            "A valid Cognito access token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    issuer = (
        f"https://cognito-idp.{settings.cognito_region}.amazonaws.com/"
        f"{settings.cognito_user_pool_id}"
    )
    try:
        signing_key = _jwks_client(issuer).get_signing_key_from_jwt(credentials.credentials)
    except PyJWKClientConnectionError as error:
        raise HTTPException(503, "Unable to verify identity provider keys") from error
    except PyJWKClientError as error:
        raise HTTPException(
            401,
            "Invalid Cognito access token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error

    try:
        claims = jwt.decode(
            credentials.credentials,
            signing_key.key,
            algorithms=["RS256"],
            issuer=issuer,
            options={
                "verify_aud": False,
                "require": ["exp", "iat", "sub", "token_use", "client_id", "iss"],
            },
        )
    except InvalidTokenError as error:
        raise HTTPException(
            401,
            "Invalid Cognito access token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error

    if (
        claims.get("token_use") != "access"
        or claims.get("client_id") != settings.cognito_app_client_id
        or not isinstance(claims.get("sub"), str)
        or not claims["sub"].strip()
    ):
        raise HTTPException(
            401,
            "Cognito access token is not authorized for this application",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return claims
