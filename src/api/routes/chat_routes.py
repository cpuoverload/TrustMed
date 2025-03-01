from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Annotated
import asyncio
import json
from api.dependencies import get_rag_engine
from api.models import ChatRequest
from rag.rag_engine import RAGEngine

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/")
async def chat(
    request: ChatRequest, rag_engine: Annotated[RAGEngine, Depends(get_rag_engine)]
):
    """Chat interface with streaming output"""
    try:

        async def generate():
            response = rag_engine.query(request.question)

            # Send the retrieved related text first
            source_text = f"Found {len(response.source_nodes)} relevant documents:\n\n"
            source_text += "----------------------------------------\n"

            for idx, node in enumerate(response.source_nodes, 1):
                # Truncate text to first 200 characters and add ellipsis if needed
                truncated_text = node.text[:200]
                if len(node.text) > 200:
                    truncated_text += "..."

                source_text += f"Document {idx}:\n"
                source_text += f"\nContent: {truncated_text}\n"
                source_text += f"\nSource Information: {node.metadata}\n\n"
                source_text += "----------------------------------------\n"

            yield f"data: {json.dumps({'delta': source_text}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)

            if hasattr(response, "response_gen"):
                for chunk in response.response_gen:
                    if isinstance(chunk, str):
                        data = chunk
                    else:
                        data = chunk.delta
                    if data:
                        yield f"data: {json.dumps({'delta': data}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.02)
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
