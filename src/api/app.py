from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import RAW_DATA_DIR, APP_PROFILE
from rag.rag_engine import create_rag_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    # 保证每个请求都使用同一个 index
    app.state.rag_engine = create_rag_engine(APP_PROFILE, RAW_DATA_DIR, streaming=True)

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
