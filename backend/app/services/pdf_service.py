"""PDF Assessment Report Generation Service for MediSense AI.

Generates a structured, professional clinical assessment report PDF
using ReportLab, populated strictly from validated assessment results.
The PDF and UI represent the exact same assessment outcome (no secondary interpretation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.models import Assessment, MedicalReport, Symptom, User, XrayResult

DISCLAIMER_TEXT = (
    "MEDICAL DISCLAIMER: MediSense AI is an academic clinical decision-support and health navigation system. "
    "This document does NOT constitute a medical diagnosis, prescription, or definitive treatment plan. "
    "All automated suggestions must be evaluated and verified by a licensed healthcare professional."
)


def generate_assessment_pdf(assessment_id: UUID | str, db: Session) -> dict[str, Any]:
    """Generate a PDF summary of the assessment grounded in validated structured results."""
    if isinstance(assessment_id, str):
        try:
            assessment_id = UUID(assessment_id)
        except ValueError:
            return {"status": "error", "message": "Invalid assessment ID format", "path": None}

    assessment = db.get(Assessment, assessment_id)
    if not assessment:
        return {"status": "error", "message": "Assessment not found", "path": None}

    user = db.get(User, assessment.user_id)
    user_name = user.name if user else "Patient"
    user_location = f"{user.district}, {user.state}" if user else "Unspecified"

    settings = get_settings()
    reports_dir = settings.upload_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_filename = f"medisense_assessment_{assessment_id}.pdf"
    pdf_path = reports_dir / pdf_filename

    # Extract feedback payload if available to guarantee exact parity with UI/API
    res_payload = assessment.result_payload or {}
    feedback = res_payload.get("feedback", {})
    triage_info = res_payload.get("triage", {})
    evidence_info = res_payload.get("evidence", {})

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom typography
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e3a8a"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4b5563"),
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e40af"),
        spaceBefore=7,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#1f2937"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#6b7280"),
    )

    story: list[Any] = []

    # Header
    story.append(Paragraph("MEDISENSE AI — CLINICAL ASSESSMENT SUMMARY", title_style))
    story.append(
        Paragraph("Deterministic Health Decision-Support & Clinical Navigation Report", subtitle_style)
    )
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a8a"), spaceAfter=10))

    # Meta strip
    pathway_color = {
        "EMERGENCY": "#dc2626",
        "CONSULTATION": "#d97706",
        "MILD": "#059669",
    }.get((assessment.pathway or "").upper(), "#4b5563")

    date_str = assessment.created_at.strftime("%B %d, %Y at %H:%M UTC")
    inputs_str = ", ".join(assessment.input_types or []).title()

    meta_data = [
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(user_name, body_style),
         Paragraph("<b>Assessment ID:</b>", body_style), Paragraph(str(assessment_id)[:16] + "...", body_style)],
        [Paragraph("<b>Location:</b>", body_style), Paragraph(user_location, body_style),
         Paragraph("<b>Assessment Date:</b>", body_style), Paragraph(date_str, body_style)],
        [Paragraph("<b>Inputs Provided:</b>", body_style), Paragraph(inputs_str, body_style),
         Paragraph("<b>Triage Pathway:</b>", body_style),
         Paragraph(f"<font color='{pathway_color}'><b>{assessment.pathway or 'PENDING'}</b></font>", body_style)],
        [Paragraph("<b>Suggested Specialty:</b>", body_style),
         Paragraph(assessment.specialty or "General Physician", body_style),
         Paragraph("<b>Rules Version:</b>", body_style), Paragraph(assessment.rules_version or "v2.1", body_style)],
    ]
    meta_table = Table(meta_data, colWidths=[110, 150, 110, 150])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 1. Assessment Summary
    story.append(Paragraph("1. Assessment Summary", section_heading))
    summary_text = feedback.get("assessment_summary") or (
        f"Multimodal assessment based on patient inputs. Assigned triage pathway: {assessment.pathway or 'PENDING'}."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 8))

    # 2. Symptoms Identified & Negative Red Flags
    story.append(Paragraph("2. Symptoms Evidence", section_heading))
    symptoms_present = feedback.get("symptoms_identified", [])
    if not symptoms_present and evidence_info:
        symptoms_present = evidence_info.get("symptoms", {}).get("present", [])
    if not symptoms_present:
        symptom_rows = db.query(Symptom).filter(Symptom.assessment_id == assessment_id).all()
        symptoms_present = [{"symptom": s.symptom, "status": s.state, "severity": s.severity, "duration": s.duration, "context": s.context} for s in symptom_rows if s.state == "PRESENT"]

    negative_flags = feedback.get("negative_red_flags", [])
    if not negative_flags and evidence_info:
        negative_flags = evidence_info.get("symptoms", {}).get("absent", [])

    if symptoms_present:
        symp_data = [["Symptom", "Status", "Duration", "Severity", "Qualifiers / Context"]]
        for s in symptoms_present:
            name_disp = s.get("symptom", "").title()
            qualifiers = []
            if s.get("laterality"):
                qualifiers.append(f"{s['laterality']}-sided")
            if s.get("quality"):
                qualifiers.append(s["quality"])
            if s.get("trigger"):
                qualifiers.append(f"Trigger: {s['trigger']}")
            if s.get("context") and s.get("context") != s.get("trigger"):
                qualifiers.append(s["context"])
            qual_str = ", ".join(qualifiers) if qualifiers else "—"

            symp_data.append(
                [
                    Paragraph(f"<b>{name_disp}</b>", body_style),
                    Paragraph(s.get("status", "PRESENT"), body_style),
                    Paragraph(s.get("duration") or "Unspecified", body_style),
                    Paragraph((s.get("severity") or "UNKNOWN").upper(), body_style),
                    Paragraph(qual_str, body_style),
                ]
            )
        symp_table = Table(symp_data, colWidths=[120, 60, 80, 80, 180])
        symp_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(symp_table)
    else:
        story.append(Paragraph("<i>No physical symptoms provided for this assessment.</i>", body_style))

    if negative_flags:
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Negative Findings / Red Flags Explicitly Denied:</b>", body_style))
        abs_items = [f"• <b>{s.get('finding') or s.get('symptom', '')}:</b> ABSENT" for s in negative_flags]
        story.append(Paragraph("<br/>".join(abs_items), body_style))
    story.append(Spacer(1, 8))

    # 3. Medical Report Findings
    story.append(Paragraph("3. Medical Report Findings", section_heading))
    report_findings = feedback.get("medical_report_findings", [])
    report_status = feedback.get("medical_report_status", "NOT_PROVIDED")
    qualitative_rep = feedback.get("qualitative_findings", []) or evidence_info.get("report", {}).get("qualitative_findings", [])
    narrative_rep = feedback.get("narrative_findings", []) or evidence_info.get("report", {}).get("narrative_findings", [])
    is_report_provided = report_status == "PROVIDED" or evidence_info.get("report", {}).get("provided", False)

    if report_findings:
        rep_data = [["Test / Parameter", "Measured Value", "Reference Range", "Status"]]
        for p in report_findings:
            interp = p.get("status") or p.get("interpretation") or "RECORDED"
            interp_color = "#dc2626" if interp in {"HIGH", "LOW"} else ("#059669" if interp == "NORMAL" else "#4b5563")
            rep_data.append(
                [
                    p.get("test_name", "Test"),
                    p.get("value", ""),
                    p.get("reference_range") or "Not provided",
                    f"<font color='{interp_color}'><b>{interp}</b></font>",
                ]
            )
        rep_table = Table([[Paragraph(c, body_style) for c in row] for row in rep_data], colWidths=[160, 120, 140, 100])
        rep_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(rep_table)

    if qualitative_rep:
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Qualitative Clinical Findings:</b>", body_style))
        q_lines = [f"• <b>{q.get('finding')}:</b> {q.get('state', 'PRESENT')}" for q in qualitative_rep]
        story.append(Paragraph("<br/>".join(q_lines), body_style))

    if narrative_rep:
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Clinical History & Narrative:</b>", body_style))
        n_lines = [f"• <b>{n.get('section')}:</b> {n.get('content')}" for n in narrative_rep]
        story.append(Paragraph("<br/>".join(n_lines), body_style))

    if not report_findings and not qualitative_rep and not narrative_rep:
        if not is_report_provided:
            story.append(Paragraph("<i>Medical Report: Not provided.</i>", body_style))
        else:
            story.append(Paragraph("<i>Medical report provided.</i>", body_style))
    story.append(Spacer(1, 8))

    # 4. X-Ray Imaging Analysis
    story.append(Paragraph("4. X-Ray Imaging Analysis", section_heading))
    xray_info = feedback.get("xray_findings", {})
    if not xray_info and evidence_info:
        xray_info = evidence_info.get("xray", {})
    if xray_info.get("received") or "xray" in (assessment.input_types or []):
        x_status = xray_info.get("status", "UNAVAILABLE")
        x_desc = xray_info.get("description") or xray_info.get("message") or "Automated interpretation unavailable."
        raw_reg = (xray_info.get("region") or "").strip().lower()
        x_reg = raw_reg.title() if raw_reg not in {"not declared", "unknown", ""} else "Not declared"
        x_pred = xray_info.get("prediction") or "None"
        x_block = (
            f"<b>Region:</b> {x_reg} &nbsp;|&nbsp; <b>Status:</b> {x_status} &nbsp;|&nbsp; "
            f"<b>Model Prediction:</b> {x_pred}<br/>"
            f"<i>{x_desc}</i>"
        )
        story.append(Paragraph(x_block, body_style))
    else:
        story.append(Paragraph("<i>X-Ray: Not provided.</i>", body_style))
    story.append(Spacer(1, 8))

    # 5. Supporting & Contradictory Evidence
    story.append(Paragraph("5. Clinical Rationale & Evidence", section_heading))
    support_items = feedback.get("supporting_evidence", [])
    if support_items:
        story.append(Paragraph("<b>Supporting Indicators:</b>", body_style))
        story.append(Paragraph("<br/>".join([f"• {item}" for item in support_items]), body_style))

    contra_items = feedback.get("contradictory_evidence", [])
    if contra_items:
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Contradictory Findings:</b>", body_style))
        story.append(Paragraph("<br/>".join([f"• <font color='#dc2626'>{item}</font>" for item in contra_items]), body_style))

    missing_items = feedback.get("uncertain_missing_info", [])
    if missing_items:
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Unknown / Unassessed Factors:</b>", body_style))
        story.append(Paragraph("<br/>".join([f"• {item}" for item in missing_items]), body_style))
    story.append(Spacer(1, 8))

    # 6. Safety Pathway & Clinical Navigation
    story.append(Paragraph("6. Safety Pathway & Recommended Next Steps", section_heading))
    why_path = feedback.get("why_this_triage_selected") or triage_info.get("why_this_pathway") or "Rule-based assessment."
    next_step_data = feedback.get("recommended_next_step", {})
    guidance = next_step_data.get("guidance") or triage_info.get("guidance") or "Consult a licensed healthcare professional."
    next_steps = next_step_data.get("next_steps") or triage_info.get("next_steps") or "Schedule an in-person medical evaluation."
    suggested_specialty = next_step_data.get("suggested_specialty") or assessment.specialty or "General Physician"

    guidance_block = [
        [Paragraph("<b>Pathway Rationale:</b>", body_style), Paragraph(why_path, body_style)],
        [Paragraph("<b>Health Guidance:</b>", body_style), Paragraph(guidance, body_style)],
        [Paragraph("<b>Suggested Specialty:</b>", body_style), Paragraph(suggested_specialty, body_style)],
        [Paragraph("<b>Recommended Next Steps:</b>", body_style), Paragraph(next_steps, body_style)],
    ]
    guidance_table = Table(guidance_block, colWidths=[140, 380])
    guidance_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(guidance_table)
    story.append(Spacer(1, 10))

    # Disclaimer Footer
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceAfter=5))
    story.append(Paragraph(DISCLAIMER_TEXT, disclaimer_style))

    # Build PDF
    doc.build(story)

    return {
        "status": "ok",
        "path": str(pdf_path),
        "filename": pdf_filename,
    }
