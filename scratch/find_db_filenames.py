import sys
from pathlib import Path
sys.path.insert(0, str(Path("backend").resolve()))
from app.database.database import SessionLocal
from app.database.models import XrayResult

db = SessionLocal()
results = db.query(XrayResult).all()
print(f"Total XrayResults in DB: {len(results)}")

seen = set()
for r in results:
    fname = r.file_name
    stored = getattr(r, "stored_file_name", None) or getattr(r, "image_path", None)
    # Check if file exists in uploads/xrays
    p = Path("backend/uploads/xrays") / (stored or "")
    if str(fname) not in seen:
        print(f"File: {str(fname):32} | Stored: {stored} | Exists: {p.exists()} | Region: {r.region}")
        seen.add(str(fname))
