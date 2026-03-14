from pydantic import BaseModel


class RetrievedDocument(BaseModel):
    text: str
    source: str
    tool: str