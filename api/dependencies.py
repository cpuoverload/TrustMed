from fastapi import Request
from api.services.rag_service import RAGService


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service
