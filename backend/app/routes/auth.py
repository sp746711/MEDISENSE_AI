"""Authentication API routes."""

from fastapi import APIRouter, HTTPException, status

from app.core.dependencies import CurrentUser, DbSession
from app.schemas.auth import (
    AuthUserResponse,
    LoginRequest,
    LoginResponse,
    RefreshTokenRequest,
    RefreshTokenResponse,
    RegisterRequest,
)
from app.schemas.user import UserResponse
from app.services import auth_service


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


@router.post(
    "/register",
    response_model=LoginResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    payload: RegisterRequest,
    db: DbSession,
) -> LoginResponse:

    try:
        user = auth_service.register_user(
            db,
            payload,
        )

    except auth_service.AuthError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc

    access_token, refresh_token = auth_service.issue_tokens_for_user(user)

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=AuthUserResponse.model_validate(user),
    )


@router.post(
    "/login",
    response_model=LoginResponse,
)
def login(
    payload: LoginRequest,
    db: DbSession,
) -> LoginResponse:

    try:
        user = auth_service.authenticate_user(
            db,
            payload,
        )

    except auth_service.AuthError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc

    access_token, refresh_token = auth_service.issue_tokens_for_user(user)

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        user=AuthUserResponse.model_validate(user),
    )


@router.post(
    "/refresh",
    response_model=RefreshTokenResponse,
)
def refresh(
    payload: RefreshTokenRequest,
    db: DbSession,
) -> RefreshTokenResponse:

    try:
        new_access_token = auth_service.refresh_access_token(
            db,
            payload.refresh_token,
        )

    except auth_service.AuthError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc

    return RefreshTokenResponse(
        access_token=new_access_token,
        token_type="bearer",
    )


@router.get(
    "/me",
    response_model=UserResponse,
)
def me(
    current_user: CurrentUser,
) -> UserResponse:

    return UserResponse.model_validate(
        current_user
    )