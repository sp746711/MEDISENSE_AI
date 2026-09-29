from pydantic import BaseModel


class FacilitySchema(BaseModel):
    status: str = "ok"
