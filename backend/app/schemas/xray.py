from pydantic import BaseModel


class XraySchema(BaseModel):
    status: str = "ok"
