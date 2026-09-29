"""Create database tables using DATABASE_URL from environment / .env."""

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.database.database import check_db_connection, init_db


def main() -> int:
    settings = get_settings()
    print(f"Using database: {settings.database_url.split('@')[-1]}")
    if not check_db_connection():
        print("ERROR: Cannot connect to the database. Set DATABASE_URL in backend/.env")
        return 1
    init_db()
    print("OK: Tables created/verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
