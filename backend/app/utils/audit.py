"""AI audit logging helpers — avoid logging sensitive clinical content."""

from typing import Any, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from app.database.models import AiAuditLog


def write_audit_log(
    db: Session,
    *,
    assessment_id: Optional[UUID],
    input_types: Optional[list] = None,
    model_versions: Optional[dict[str, Any]] = None,
    nlp_version: Optional[str] = None,
    ocr_version: Optional[str] = None,
    rules_version: Optional[str] = None,
    llm_version: Optional[str] = None,
    knowledge_source_ids: Optional[list] = None,
    output_status: Optional[str] = None,
) -> AiAuditLog:
    entry = AiAuditLog(
        assessment_id=assessment_id,
        input_types=input_types,
        model_versions=model_versions,
        nlp_version=nlp_version,
        ocr_version=ocr_version,
        rules_version=rules_version,
        llm_version=llm_version,
        knowledge_source_ids=knowledge_source_ids,
        output_status=output_status,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry
