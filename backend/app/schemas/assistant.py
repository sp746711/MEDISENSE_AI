from pydantic import BaseModel


class AssistantSchema(BaseModel):
    status: str = "ok"
