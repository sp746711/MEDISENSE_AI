"""Health and system status routes."""

from fastapi import APIRouter

from app.core.config import get_settings
from app.database.database import check_db_connection

router = APIRouter(tags=["Health"])


@router.get("/health")
def health() -> dict:
    """Public health check used by frontend and ops."""
    settings = get_settings()
    db_ok = check_db_connection()
    return {
        "status": "ok" if db_ok else "degraded",
        "app": settings.app_name,
        "environment": settings.app_env,
        "database": "connected" if db_ok else "unavailable",
        "message": (
            "MediSense AI API is running"
            if db_ok
            else "API is running but database is unavailable. Check DATABASE_URL."
        ),
    }
