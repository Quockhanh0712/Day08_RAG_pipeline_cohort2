"""RAG Retrieval Agent entry point — port 10101."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import uvicorn
from dotenv import load_dotenv

# Ensure we can import modules from parent folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from common.registry_client import register
from rag_retrieval_agent.agent_executor import RAGRetrievalAgentExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [rag_retrieval_agent] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PORT = 10101
AGENT_ENDPOINT = f"http://localhost:{PORT}"


async def _register_with_retry(max_attempts: int = 15, delay: float = 2.0) -> None:
    """Retry registration until the registry is up."""
    info = {
        "agent_name": "rag-retrieval-agent",
        "version": "1.0",
        "description": "Specialist retrieval agent using Hybrid Search and Jina Rerank",
        "tasks": ["rag_retrieval"],
        "endpoint": AGENT_ENDPOINT,
        "tags": ["retrieval", "legal-rag", "search"],
    }
    for attempt in range(1, max_attempts + 1):
        try:
            await register(info)
            logger.info("Registered with registry (attempt %d)", attempt)
            return
        except Exception as exc:
            logger.warning(
                "Registry not ready (attempt %d/%d): %s — retrying in %.0fs",
                attempt, max_attempts, exc, delay,
            )
            await asyncio.sleep(delay)
    logger.error("Failed to register after %d attempts", max_attempts)


async def main() -> None:
    # Trigger registry registration in the background
    asyncio.create_task(_register_with_retry())

    agent_card = AgentCard(
        name="RAG Retrieval Agent",
        description=(
            "Retrieves context documents from Vector Store (Weaviate) using Hybrid Search "
            "and Reranks results using Jina AI."
        ),
        url=AGENT_ENDPOINT,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="rag_retrieval",
                name="RAG Retrieval",
                description="Performs dense/sparse hybrid search and cross-encoder reranking.",
                tags=["retrieval", "weaviate", "rerank"],
            )
        ],
    )

    executor = RAGRetrievalAgentExecutor()
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )
    app_builder = A2AFastAPIApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    app = app_builder.build()

    # Enable CORS
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="info")
    server = uvicorn.Server(config)
    logger.info("RAG Retrieval Agent listening on port %d", PORT)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
