"""Generation Agent server entry point — port 10102."""

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
from generation_agent.agent_executor import GenerationAgentExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [generation_agent] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

PORT = 10102
AGENT_ENDPOINT = f"http://localhost:{PORT}"


async def _register_with_retry(max_attempts: int = 15, delay: float = 2.0) -> None:
    """Retry registration until the registry is up."""
    info = {
        "agent_name": "generation-agent",
        "version": "1.0",
        "description": "Specialist generation agent answering legal questions with citations",
        "tasks": ["legal_generation"],
        "endpoint": AGENT_ENDPOINT,
        "tags": ["generation", "legal-rag", "citations"],
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
        name="Generation Agent",
        description=(
            "Generates comprehensive legal answers in Vietnamese using retrieved context, "
            "with precise citations matching the original source style."
        ),
        url=AGENT_ENDPOINT,
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="legal_generation",
                name="Legal Generation",
                description="Generates legally accurate answers based on dynamic retrieved context.",
                tags=["generation", "citations", "vietnamese"],
            )
        ],
    )

    executor = GenerationAgentExecutor()
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
    logger.info("Generation Agent listening on port %d", PORT)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
