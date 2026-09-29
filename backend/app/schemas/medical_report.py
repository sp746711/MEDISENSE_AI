from pydantic import BaseModel


class MedicalReportSchema(BaseModel):
    status: str = "ok"
