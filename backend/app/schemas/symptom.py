"""Additional schemas — expanded in later stages."""

from pydantic import BaseModel


class PlaceholderSchema(BaseModel):
    status: str = "ok"
