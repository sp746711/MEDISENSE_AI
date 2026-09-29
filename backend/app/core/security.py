"""Security helpers: password hashing and JWT tokens."""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a plaintext password for storage."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    """Verify a plaintext password against a stored hash."""
    return pwd_context.verify(plain_password, password_hash)


def create_access_token(
    subject: str,
    expires_minutes: Optional[int] = None,
    extra_claims: Optional[dict[str, Any]] = None,
) -> str:
    """Create a signed JWT access token."""
    settings = get_settings()
    expire_delta = expires_minutes or settings.access_token_expire_minutes
    expire = datetime.now(timezone.utc) + timedelta(minutes=expire_delta)
    payload: dict[str, Any] = {"sub": subject, "exp": expire}
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate a JWT. Raises JWTError on failure."""
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.algorithm])


class TokenError(Exception):
    """Raised when a token cannot be validated."""


def get_subject_from_token(token: str) -> str:
    """Extract the subject (user id) from a JWT."""
    try:
        payload = decode_access_token(token)
        subject = payload.get("sub")
        if not subject:
            raise TokenError("Token missing subject")
        return str(subject)
    except JWTError as exc:
        raise TokenError("Invalid or expired token") from exc
