"""SQLAlchemy engine and session factory."""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def init_db() -> None:
    """Create all tables. Safe to call on startup."""
    # Import models so metadata is populated
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _migrate_schema()


def _migrate_schema() -> None:
    """Safely apply non-destructive schema adjustments if columns are missing."""
    try:
        with engine.begin() as conn:
            # Check for report_category in medical_reports
            try:
                conn.execute(text("SELECT report_category FROM medical_reports LIMIT 1"))
            except Exception:
                try:
                    conn.execute(
                        text("ALTER TABLE medical_reports ADD COLUMN report_category VARCHAR(50) DEFAULT 'medical_report' NOT NULL")
                    )
                except Exception:
                    pass

            # Check for file_name in xray_results
            try:
                conn.execute(text("SELECT file_name FROM xray_results LIMIT 1"))
            except Exception:
                try:
                    conn.execute(
                        text("ALTER TABLE xray_results ADD COLUMN file_name VARCHAR(255)")
                    )
                except Exception:
                    pass
    except Exception:
        pass


def check_db_connection() -> bool:
    """Return True if the database accepts a simple query."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
