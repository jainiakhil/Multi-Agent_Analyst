"""FastAPI application entry point.

Exposes REST endpoints for document ingestion and codebase analysis,
along with real-time WebSocket streaming of LangGraph agent events.
"""

import os
import shutil
import logging
from typing import Optional
from pathlib import Path

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    HTTPException,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.core.config import settings
from app.graph.workflow import compiled_graph
from app.services.vector_store import get_vector_store_service

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("analyst_api")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.API_VERSION,
    description="Multi-Agent Code & Document Analysis Engine powered by LangGraph, Ollama, and ChromaDB.",
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    """Payload schema for synchronous analysis queries."""

    query: str = Field(..., description="Natural language analysis request or question.")
    thread_id: Optional[str] = Field(
        default="session_default",
        description="Persistent conversation session / checkpoint identifier.",
    )
    target_file: Optional[str] = Field(
        default=None,
        description="Optional local file path to audit via Python AST.",
    )


class AnalyzeResponse(BaseModel):
    """Response schema for synchronous analysis queries."""

    status: str
    thread_id: str
    final_response: str
    messages: list[dict]


@app.get("/health", tags=["System"])
@app.get("/api/v1/health", tags=["System"])
async def health_check():
    """Health check endpoint confirming API availability and runtime configuration."""
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "ollama_base_url": settings.OLLAMA_BASE_URL,
        "supervisor_model": settings.SUPERVISOR_MODEL,
        "code_analyst_model": settings.CODE_ANALYST_MODEL,
        "doc_parser_model": settings.DOC_PARSER_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
    }


@app.post("/api/v1/documents/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """Ingests a technical specification or documentation file into ChromaDB vector storage.

    Supported formats: `.pdf`, `.txt`, `.md`.
    """
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = Path(file.filename or "").suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {list(allowed_extensions)}",
        )

    upload_dir = Path("./data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / (file.filename or "uploaded_document")

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        service = get_vector_store_service()
        if file_ext == ".pdf":
            chunks_indexed = service.ingest_pdf(str(temp_path))
        else:
            chunks_indexed = service.ingest_text_file(str(temp_path))

        logger.info(f"Indexed {chunks_indexed} chunks for file: {file.filename}")
        return {
            "status": "success",
            "chunks_indexed": chunks_indexed,
            "filename": file.filename,
        }
    except Exception as exc:
        logger.error(f"Failed to ingest document '{file.filename}': {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(exc)}",
        )
    finally:
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.post("/api/v1/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_endpoint(payload: AnalyzeRequest):
    """Performs a synchronous multi-agent analysis cycle and returns the completed state."""
    config = {"configurable": {"thread_id": payload.thread_id}}
    inputs = {
        "messages": [HumanMessage(content=payload.query)],
        "target_file": payload.target_file,
    }

    try:
        final_state = await compiled_graph.ainvoke(inputs, config=config)
        messages_out = []
        for msg in final_state.get("messages", []):
            messages_out.append({"type": msg.type, "content": msg.content})

        last_msg = messages_out[-1]["content"] if messages_out else "No response generated."
        return AnalyzeResponse(
            status="completed",
            thread_id=payload.thread_id,
            final_response=last_msg,
            messages=messages_out,
        )
    except Exception as exc:
        logger.error(f"Analysis invocation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline error: {str(exc)}",
        )


@app.websocket("/ws/analyze")
async def websocket_analyze_endpoint(websocket: WebSocket):
    """Provides real-time streaming graph node updates and agent responses over WebSockets."""
    await websocket.accept()
    logger.info("WebSocket client connected to /ws/analyze")

    try:
        data = await websocket.receive_json()
        user_query = data.get("query")
        if not user_query:
            await websocket.send_json({"error": "Missing required field 'query'"})
            await websocket.close()
            return

        thread_id = data.get("thread_id", "session_default")
        target_file = data.get("target_file")

        config = {"configurable": {"thread_id": thread_id}}
        inputs = {
            "messages": [HumanMessage(content=user_query)],
            "target_file": target_file,
        }

        # Stream node execution events to WebSocket client
        async for event in compiled_graph.astream(inputs, config=config):
            for node_name, state_update in event.items():
                latest_content = None
                if "messages" in state_update and state_update["messages"]:
                    latest_content = state_update["messages"][-1].content

                payload = {
                    "node": node_name,
                    "next_node": state_update.get("next_node", ""),
                    "latest_message": latest_content,
                }
                await websocket.send_json(payload)

        await websocket.send_json({"status": "completed"})
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as exc:
        logger.error(f"WebSocket execution error: {exc}")
        try:
            await websocket.send_json({"error": str(exc)})
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
