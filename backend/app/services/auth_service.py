"""Authentication service: register, login and token issuing."""

from sqlalchemy.orm import Session

from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from app.database.models import User
from app.schemas.auth import LoginRequest, RegisterRequest


class AuthError(Exception):
    """Application-level authentication error."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
    ):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def register_user(
    db: Session,
    payload: RegisterRequest,
) -> User:
    """Register a new user."""

    email = str(payload.email).strip().lower()

    existing = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if existing:
        raise AuthError(
            "An account with this email already exists.",
            status_code=409,
        )

    user = User(
        name=payload.name.strip(),
        email=email,
        password_hash=hash_password(payload.password),
        state=payload.state.strip(),
        district=payload.district.strip(),
        city=payload.city.strip() if payload.city else None,
    )

    db.add(user)

    try:
        db.commit()
        db.refresh(user)

    except Exception:
        db.rollback()
        raise

    return user


def authenticate_user(
    db: Session,
    payload: LoginRequest,
) -> User:
    """Authenticate user by email and password."""

    email = str(payload.email).strip().lower()

    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if user is None:
        raise AuthError(
            "Invalid email or password.",
            status_code=401,
        )

    if not verify_password(
        payload.password,
        user.password_hash,
    ):
        raise AuthError(
            "Invalid email or password.",
            status_code=401,
        )

    return user


def issue_token_for_user(
    user: User,
) -> str:
    """Create JWT containing the user's UUID."""

    if not user.user_id:
        raise AuthError(
            "User ID is missing.",
            status_code=500,
        )

    return create_access_token(
        subject=str(user.user_id),
    )