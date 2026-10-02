"""Security helpers: password hashing and JWT authentication."""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings


pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


class TokenError(Exception):
    """Raised when a JWT cannot be validated."""


def hash_password(password: str) -> str:
    """Hash a plaintext password."""
    if not password:
        raise ValueError("Password cannot be empty")

    return pwd_context.hash(password)


def verify_password(
    plain_password: str,
    password_hash: str,
) -> bool:
    """Verify plaintext password against stored bcrypt hash."""
    try:
        return pwd_context.verify(
            plain_password,
            password_hash,
        )
    except Exception:
        return False


def create_access_token(
    subject: str,
    expires_minutes: Optional[int] = None,
    extra_claims: Optional[dict[str, Any]] = None,
) -> str:
    """Create a signed JWT access token."""

    settings = get_settings()

    minutes = (
        expires_minutes
        if expires_minutes is not None
        else settings.access_token_expire_minutes
    )

    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=minutes)

    payload: dict[str, Any] = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": "access",
    }

    if extra_claims:
        payload.update(extra_claims)

    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.algorithm,
    )


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate JWT."""

    if not token or not isinstance(token, str):
        raise TokenError("Token is missing")

    settings = get_settings()

    try:
        payload = jwt.decode(
            token.strip(),
            settings.jwt_secret_key,
            algorithms=[settings.algorithm],
        )
    except JWTError as exc:
        raise TokenError("Invalid or expired token") from exc
    except Exception as exc:
        raise TokenError("Unable to validate token") from exc

    if not payload:
        raise TokenError("Empty token payload")

    if payload.get("type") != "access":
        raise TokenError("Invalid token type")

    subject = payload.get("sub")

    if not subject:
        raise TokenError("Token missing subject")

    return payload


def get_subject_from_token(token: str) -> str:
    """Return the user ID stored in the JWT subject."""

    payload = decode_access_token(token)

    subject = payload.get("sub")

    if not subject:
        raise TokenError("Token missing subject")

    return str(subject)