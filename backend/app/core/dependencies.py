"""FastAPI dependency providers.

Provides:
- Database sessions
- Current authenticated user from JWT
"""

from typing import Annotated, Generator
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import TokenError, get_subject_from_token
from app.database.database import SessionLocal
from app.database.models import User


bearer_scheme = HTTPBearer(
    auto_error=False,
)


def get_db() -> Generator[Session, None, None]:
    """Create and safely close a SQLAlchemy session."""

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


DbSession = Annotated[
    Session,
    Depends(get_db),
]


def get_current_user(
    db: DbSession,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
) -> User:
    """Resolve authenticated user from Bearer JWT."""

    # ---------------------------------------------------------
    # 1. Check Authorization header
    # ---------------------------------------------------------

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials.strip()

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is empty.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ---------------------------------------------------------
    # 2. Decode JWT
    # ---------------------------------------------------------

    try:
        user_id = get_subject_from_token(token)

    except TokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please login again.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    # ---------------------------------------------------------
    # 3. Validate UUID
    # ---------------------------------------------------------

    try:
        user_uuid = UUID(user_id)

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID in token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    # ---------------------------------------------------------
    # 4. Find user
    # ---------------------------------------------------------

    user = db.get(User, user_uuid)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User associated with this token no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


CurrentUser = Annotated[
    User,
    Depends(get_current_user),
]