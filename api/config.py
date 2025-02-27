from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag.chroma import load_or_create_chroma_index
from api.services.rag_service import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    index = load_or_create_chroma_index()
    app.state.rag_service = RAGService(index, streaming=True)

    yield

    # cleanup


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Chat API", version="1.0.0", lifespan=lifespan)

    # configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app
