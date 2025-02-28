from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root():
    """Health check interface"""
    return {"status": "ok", "message": "RAG Chat API is running"}
