import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
import json
from fastapi.testclient import TestClient
from app.main import app
from app.database.database import SessionLocal
from app.database.models import User, Assessment
from app.core.dependencies import get_current_user
from app.ai.evaluate_fullbody_dataset import unique_entries

def test_live_api():
    db = SessionLocal()
    test_user = db.query(User).first()
    if not test_user:
        test_user = User(email="test@example.com", hashed_password="pw", is_active=True)
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        
    test_assessment = db.query(Assessment).filter(Assessment.user_id == test_user.user_id).first()
    if not test_assessment:
        test_assessment = Assessment(user_id=test_user.user_id, input_types=["xray"], status="draft")
        db.add(test_assessment)
        db.commit()
        db.refresh(test_assessment)
    else:
        test_assessment.input_types = ["xray"]
        db.commit()

    app.dependency_overrides[get_current_user] = lambda: test_user
    client = TestClient(app)
    
    target_names = [
        "05_elbow.png",
        "12_elbow (1).png",
        "09_ankle.png",
        "13_forearm.png",
        "07_hand.png",
        "04_shoulder.png",
        "08_knee.png",
        "01_chest.png",
        "02_skull.png",
        "03_cervical_spine.png",
        "10_abdomen_kub.png",
        "17_abdomen_kub.png",
    ]
    
    print("\n--- Testing Live API Endpoint: POST /api/assessments/{assessment_id}/xray ---\n")
    
    results = []
    seen = set()
    for entry in unique_entries:
        filename = entry.get("filename", "")
        matched_target = None
        for t in target_names:
            if t in filename and t not in seen:
                matched_target = t
                break
                
        if not matched_target:
            continue
            
        img_path = Path(entry["path"])
        if not img_path.exists():
            continue
            
        seen.add(matched_target)
        file_bytes = img_path.read_bytes()
        
        response = client.post(
            f"/api/assessments/{test_assessment.assessment_id}/xray",
            files={"file": (matched_target, file_bytes, "image/png")},
        )
        
        assert response.status_code == 200, f"Failed for {matched_target}: {response.text}"
        data = response.json()
        
        # Verify structure
        p_pred = data.get("primary_prediction") or {}
        s_pred = data.get("specialist_prediction") or {}
        comp = data.get("two_model_comparison") or {}
        final_cand = data.get("final_candidate")
        sel_source = data.get("selection_source")
        
        # Assert required fields exist
        assert "candidate_body_part" in p_pred
        assert "confidence" in p_pred or "body_part_confidence" in p_pred
        assert "top_predictions" in p_pred
        assert "decision_case" in comp or "comparison_case" in comp
        assert "source_attributed_confidence" in comp or "attributed_confidence" in comp
        
        p_conf = (p_pred.get('confidence') or p_pred.get('body_part_confidence') or 0.0) * 100
        s_conf = (s_pred.get('confidence') or s_pred.get('body_part_confidence') or 0.0) * 100
        s_cand = s_pred.get('candidate_body_part') or 'None'
        s_stat = s_pred.get('status') or 'None'
        comp_case = comp.get('decision_case') or comp.get('comparison_case', 'None')
        
        print(f"| `{matched_target}` | Primary: `{p_pred.get('candidate_body_part')}` ({p_conf:.1f}%) | "
              f"Specialist: `{s_cand}` ({s_conf:.1f}% | {s_stat}) | "
              f"Case: `{comp_case}` | Final: `{final_cand}` | Source: `{sel_source}` |")
              
        results.append({
            "target": matched_target,
            "primary": f"{p_pred.get('candidate_body_part')} ({p_conf:.1f}%)",
            "specialist": f"{s_cand} ({s_conf:.1f}% | {s_stat})",
            "case": comp_case,
            "final": final_cand,
            "source": sel_source
        })

    print(f"\n{len(results)} live benchmark images successfully executed through API and verified!")

if __name__ == "__main__":
    test_live_api()
