import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[2]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.database.database import SessionLocal
from app.database.models import XrayResult

db = SessionLocal()
files_to_check = ['05_elbow.png', '12_elbow (1).png', '09_ankle.png', '12_elbow.png']

print("=" * 80)
print("DATABASE AUDIT OF TEST IMAGES")
print("=" * 80)

for fname in files_to_check:
    rows = db.query(XrayResult).filter(XrayResult.file_name == fname).order_by(XrayResult.created_at.desc()).all()
    print(f"\n--- Records for {fname} (Total: {len(rows)}) ---")
    for r in rows:
        meta = r.anatomy_metadata or {}
        clf_out = meta.get("classifier_output") or {}
        ref = clf_out.get("upper_extremity_refinement") or {}
        print(f"ID: {r.xray_result_id} | Created: {r.created_at} | Stored: {r.stored_name}")
        print(f"  Region: {r.region} | Candidate: {meta.get('candidate_region')} | Conf: {meta.get('body_part_confidence')}")
        print(f"  Gate Passed: {meta.get('classifier_gate_passed')} | Confirmed: {getattr(r, 'anatomy_user_confirmed', None)}")
        print(f"  Specialist Refinement: status={ref.get('status')}, cand={ref.get('candidate_body_part')}, conf={ref.get('confidence')}")
