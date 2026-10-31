import sys
from pathlib import Path
sys.path.insert(0, str(Path("backend").resolve()))
from app.database.database import SessionLocal
from app.database.models import XrayResult

db = SessionLocal()
results = db.query(XrayResult).order_by(XrayResult.created_at.desc()).all()
print(f"Total XrayResults in DB: {len(results)}")

seen = {}
for r in results:
    fname = r.file_name
    stored = r.stored_name
    if fname and stored and fname not in seen:
        p = Path("backend/uploads/xrays") / stored
        seen[fname] = (stored, p.exists(), r.region)

print(f"Unique files with stored_name: {len(seen)}")
for fname, (stored, exists, reg) in sorted(seen.items()):
    print(f"File: {fname:36} | Stored: {stored} | Exists: {exists} | Region: {reg}")
