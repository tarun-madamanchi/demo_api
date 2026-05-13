from pydantic import BaseModel


class SampleDTO(BaseModel):
    user_id: str
    name: str
    description: str | None = None
