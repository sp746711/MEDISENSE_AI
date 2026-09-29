"""Authentication service: register, login, me."""

from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password, verify_password
from app.database.models import User
from app.schemas.auth import LoginRequest, RegisterRequest


class AuthError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def register_user(db: Session, payload: RegisterRequest) -> User:
    existing = db.query(User).filter(User.email == payload.email.lower()).first()
    if existing:
        raise AuthError("An account with this email already exists", status_code=409)

    user = User(
        name=payload.name.strip(),
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        state=payload.state.strip(),
        district=payload.district.strip(),
        city=payload.city.strip() if payload.city else None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, payload: LoginRequest) -> User:
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise AuthError("Invalid email or password", status_code=401)
    return user


def issue_token_for_user(user: User) -> str:
    return create_access_token(subject=str(user.user_id))
