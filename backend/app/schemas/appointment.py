from pydantic import BaseModel


class AppointmentSchema(BaseModel):
    status: str = "ok"
