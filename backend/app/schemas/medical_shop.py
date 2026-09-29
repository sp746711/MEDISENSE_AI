from pydantic import BaseModel


class MedicalShopSchema(BaseModel):
    status: str = "ok"
