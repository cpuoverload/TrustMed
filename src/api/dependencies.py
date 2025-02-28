from fastapi import Request
from rag.rag_engine import RAGEngine


def get_rag_engine(request: Request) -> RAGEngine:
    return request.app.state.rag_engine
