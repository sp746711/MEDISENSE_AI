"""PDF Assessment Report Generation Service for MediSense AI.

Generates a structured, professional clinical assessment report PDF
using ReportLab, populated strictly from real database records.
Never invents data.
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
    "MEDICAL DISCLAIMER: MediSense AI is an academic decision-support and health navigation system. "
    "This document does NOT constitute a clinical diagnosis, medical prescription, or definitive treatment plan. "
    "Automated findings must be reviewed and verified by a licensed healthcare professional."
)


def generate_assessment_pdf(assessment_id: UUID | str, db: Session) -> dict[str, Any]:
    """Generate a PDF summary of the assessment and return the file path."""
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

    # Query associated records
    symptoms = (
        db.query(Symptom)
        .filter(Symptom.assessment_id == assessment_id)
        .all()
    )
    reports = (
        db.query(MedicalReport)
        .filter(MedicalReport.assessment_id == assessment_id)
        .all()
    )
    xrays = (
        db.query(XrayResult)
        .filter(XrayResult.assessment_id == assessment_id)
        .all()
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e3a8a"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4b5563"),
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e40af"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1f2937"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#6b7280"),
    )

    story: list[Any] = []

    # Header
    story.append(Paragraph("MEDISENSE AI — HEALTH ASSESSMENT REPORT", title_style))
    story.append(
        Paragraph("Multimodal Health Assessment and Healthcare Navigation Summary", subtitle_style)
    )
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a8a"), spaceAfter=12))

    # Assessment & Patient Details
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
         Paragraph("<b>Status:</b>", body_style), Paragraph(assessment.status.upper(), body_style)],
    ]
    meta_table = Table(meta_data, colWidths=[110, 150, 110, 150])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Section 1: Symptoms
    story.append(Paragraph("1. Symptoms Evidence", section_heading))
    if symptoms:
        symp_data = [["Symptom", "State", "Duration", "Severity", "Body Area"]]
        for s in symptoms:
            symp_data.append(
                [
                    s.symptom.title(),
                    s.state,
                    s.duration or "Unspecified",
                    (s.severity or "Unspecified").title(),
                    (s.body_area or "General").title(),
                ]
            )
        symp_table = Table(symp_data, colWidths=[130, 80, 110, 90, 110])
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
        story.append(Paragraph("<i>No symptoms were provided for this assessment.</i>", body_style))
    story.append(Spacer(1, 10))

    # Section 2: Medical Report Analysis
    story.append(Paragraph("2. Medical Report Findings", section_heading))
    if reports:
        has_findings = False
        rep_data = [["Test / Parameter", "Measured Value", "Reference Range", "Interpretation"]]
        for r in reports:
            struct = r.structured_findings or {}
            params = struct.get("lab_parameters", [])
            for p in params:
                has_findings = True
                unit = p.get("unit") or ""
                val_display = f"{p.get('value')} {unit}".strip()
                interp = p.get("interpretation") or "RECORDED"
                interp_color = "#dc2626" if interp in {"HIGH", "LOW"} else "#059669"
                rep_data.append(
                    [
                        p.get("test_name", "Test"),
                        val_display,
                        p.get("reference_range") or "Not provided",
                        f"<font color='{interp_color}'><b>{interp}</b></font>",
                    ]
                )
        if has_findings:
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
        else:
            story.append(Paragraph("<i>Medical report uploaded; no structured quantitative values detected.</i>", body_style))
    else:
        story.append(Paragraph("<i>No medical report uploaded for this assessment.</i>", body_style))
    story.append(Spacer(1, 10))

    # Section 3: X-Ray Analysis
    story.append(Paragraph("3. X-Ray Imaging Analysis", section_heading))
    if xrays:
        for x in xrays:
            xray_info = (
                f"<b>Region:</b> {(x.region or 'General').title()} &nbsp;|&nbsp; "
                f"<b>Status:</b> {x.status.upper()} &nbsp;|&nbsp; "
                f"<b>Model:</b> {x.model_version or 'N/A'}<br/>"
                f"<b>Finding:</b> {x.prediction or 'No automated prediction generated'} &nbsp;|&nbsp; "
                f"<b>Uncertainty:</b> {x.uncertainty or 'Clinical correlation needed'}<br/>"
                f"<i>{x.message or ''}</i>"
            )
            story.append(Paragraph(xray_info, body_style))
    else:
        story.append(Paragraph("<i>No X-ray image uploaded for this assessment.</i>", body_style))
    story.append(Spacer(1, 10))

    # Section 4: Safety Pathway & Next Steps
    story.append(Paragraph("4. Safety Pathway & Clinical Navigation", section_heading))
    res_payload = assessment.result_payload or {}
    triage_info = res_payload.get("triage", {})
    why_path = triage_info.get("why_this_pathway") or "Rule-based assessment."
    guidance = triage_info.get("guidance") or "Consult a licensed healthcare professional for evaluation."
    next_steps = triage_info.get("next_steps") or "Schedule an in-person doctor consultation."

    guidance_block = [
        [Paragraph("<b>Pathway Rationale:</b>", body_style), Paragraph(why_path, body_style)],
        [Paragraph("<b>Health Guidance:</b>", body_style), Paragraph(guidance, body_style)],
        [Paragraph("<b>Recommended Next Steps:</b>", body_style), Paragraph(next_steps, body_style)],
    ]
    guidance_table = Table(guidance_block, colWidths=[140, 380])
    guidance_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(guidance_table)
    story.append(Spacer(1, 14))

    # Disclaimer Footer
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
    story.append(Paragraph(DISCLAIMER_TEXT, disclaimer_style))

    # Build document
    doc.build(story)

    return {
        "status": "ok",
        "path": str(pdf_path),
        "filename": pdf_filename,
    }
