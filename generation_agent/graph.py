"""Generation Agent LangGraph definition."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

# Ensure parent folder imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.llm import get_llm

logger = logging.getLogger(__name__)

# State definition
class GenerationState(TypedDict):
    question: str
    trace_id: str
    context_id: str
    depth: int
    context_json: str
    final_answer: str

# Reorder to avoid lost-in-the-middle
def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    if len(chunks) <= 2:
        return chunks

    reordered = [None] * len(chunks)
    left = 0
    right = len(chunks) - 1
    
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            reordered[left] = chunk
            left += 1
        else:
            reordered[right] = chunk
            right -= 1
            
    return reordered

# Format context for prompt
def format_context(chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)

# Node 1: Call Retrieval Agent via A2A
async def retrieve_node(state: GenerationState) -> dict:
    from common.a2a_client import delegate
    from common.registry_client import discover

    question = state["question"]
    trace_id = state["trace_id"]
    context_id = state["context_id"]
    depth = state["depth"]

    logger.info("Generation Agent: Discovering and calling RAG Retrieval Agent...")
    try:
        endpoint = await discover("rag_retrieval")
        # Call retrieval agent
        retrieved_text = await delegate(
            endpoint=endpoint,
            question=question,
            context_id=context_id,
            trace_id=trace_id,
            depth=depth + 1,
        )
        return {"context_json": retrieved_text}
    except Exception as exc:
        logger.exception("Failed to contact RAG Retrieval Agent: %s", exc)
        return {"context_json": "[]"}

# Node 2: Call LLM to generate response with citations
async def generate_node(state: GenerationState) -> dict:
    question = state["question"]
    context_json = state.get("context_json", "[]")

    try:
        chunks = json.loads(context_json)
    except Exception:
        logger.warning("Could not parse context_json as JSON list, defaulting to empty list.")
        chunks = []

    # Reorder and format context
    reordered_chunks = reorder_for_llm(chunks)
    context_str = format_context(reordered_chunks)

    # Inject context into user prompt
    user_message = f"Context:\n{context_str}\n\n---\n\nQuestion: {question}"

    system_prompt = (
        "Answer the following question comprehensively in Vietnamese.\n"
        "For every statement of fact or claim, immediately insert a citation in brackets "
        "linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3] "
        "or [VnExpress, 2024]).\n\n"
        "If the information is not explicitly stated in the provided context or knowledge "
        "base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than "
        "guessing.\n\n"
        "Rules:\n"
        "- Only use information from the provided context\n"
        "- Every factual claim MUST have a citation\n"
        "- If context is insufficient, say so clearly\n"
        "- Structure your answer with clear paragraphs"
    )

    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    logger.info("Calling LLM to generate final Vietnamese answer...")
    response = await llm.ainvoke(messages)
    
    return {"final_answer": response.content}

def create_graph() -> Any:
    workflow = StateGraph(GenerationState)

    workflow.add_node("retrieve_context", retrieve_node)
    workflow.add_node("generate_answer", generate_node)

    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()
