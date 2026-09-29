"""MediSense AI — FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.database.database import init_db
from app.routes import (
    appointments,
    assessments,
    assistant,
    auth,
    dashboard,
    facilities,
    health,
    history,
    medical_reports,
    medical_shops,
    providers,
    symptoms,
    users,
    xray,
)
from app.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    (settings.upload_path / "reports").mkdir(parents=True, exist_ok=True)
    (settings.upload_path / "xrays").mkdir(parents=True, exist_ok=True)
    (settings.upload_path / "explainability").mkdir(parents=True, exist_ok=True)

    try:
        init_db()
        logger.info("Database tables initialized")
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Database initialization failed — API will start in degraded mode: %s",
            exc,
        )

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        description=(
            "MediSense AI: Multimodal AI-Powered Health Assessment and "
            "Healthcare Navigation System (academic decision-support)."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = settings.api_prefix.rstrip("/") or "/api"

    application.include_router(health.router, prefix=prefix)
    application.include_router(auth.router, prefix=prefix)
    application.include_router(users.router, prefix=prefix)
    application.include_router(dashboard.router, prefix=prefix)
    application.include_router(assessments.router, prefix=prefix)
    application.include_router(symptoms.router, prefix=prefix)
    application.include_router(medical_reports.router, prefix=prefix)
    application.include_router(xray.router, prefix=prefix)
    application.include_router(providers.router, prefix=prefix)
    application.include_router(facilities.router, prefix=prefix)
    application.include_router(medical_shops.router, prefix=prefix)
    application.include_router(appointments.router, prefix=prefix)
    application.include_router(history.router, prefix=prefix)
    application.include_router(assistant.router, prefix=prefix)

    @application.get("/")
    def root() -> dict:
        return {
            "app": settings.app_name,
            "message": "MediSense AI API",
            "docs": "/docs",
            "health": f"{prefix}/health",
        }

    return application


app = create_app()
