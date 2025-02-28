from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    question: str
    stream: Optional[bool] = True


class ChatResponse(BaseModel):
    answer: str
