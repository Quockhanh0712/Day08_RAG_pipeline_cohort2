"""FastAPI backend for the group RAG chatbot."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.task10_generation import generate_with_citation


class ChatMessage(BaseModel):
    """Message structure for history."""

    role: str = Field(..., description="Role of message author (user or assistant)")
    content: str = Field(..., description="Text content of the message")


class ChatRequest(BaseModel):
    """Request body for chatbot questions."""

    message: str = Field(..., min_length=1, description="User question")
    history: list[ChatMessage] = Field(default_factory=list, description="Conversation history")
    top_k: int = Field(default=8, ge=1, le=10, description="Number of context chunks to retrieve")


class SourceChunk(BaseModel):
    """Sanitized source chunk returned to the web UI."""

    content: str
    score: float | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Response body for chatbot answers."""

    answer: str
    retrieval_source: str
    sources: list[SourceChunk]


app = FastAPI(
    title="Drug Law RAG Chatbot API",
    description="HTTP wrapper around the Day 8 RAG pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sanitize_sources(sources: list[dict], max_chars: int = 700) -> list[SourceChunk]:
    """Limit source payload size for browser rendering."""
    sanitized = []
    for item in sources:
        content = " ".join(str(item.get("content", "")).split())
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "..."

        score = item.get("score")
        sanitized.append(
            SourceChunk(
                content=content,
                score=float(score) if score is not None else None,
                source=item.get("source"),
                metadata=item.get("metadata", {}) or {},
            )
        )
    return sanitized


@app.get("/health")
def health() -> dict[str, str]:
    """Simple readiness endpoint for the Node.js UI."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Generate a cited RAG answer for one user message."""
    try:
        history_list = [{"role": msg.role, "content": msg.content} for msg in request.history]
        result = generate_with_citation(request.message, history=history_list, top_k=request.top_k)
    except Exception as exc:  # Keep API errors readable for the UI.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        answer=result.get("answer", ""),
        retrieval_source=result.get("retrieval_source", "none"),
        sources=_sanitize_sources(result.get("sources", [])),
    )
