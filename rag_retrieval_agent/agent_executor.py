"""RAG Retrieval Agent — AgentExecutor bridge between A2A SDK and retrieval pipeline."""

from __future__ import annotations

import json
import logging
import os
import sys
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart

# Ensure we can import src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.task9_retrieval_pipeline import retrieve

logger = logging.getLogger(__name__)

class RAGRetrievalAgentExecutor(AgentExecutor):
    """Bridges A2A RequestContext to the RAG Retrieval Agent logic."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        question = self._extract_question(context)
        context_id = context.context_id or str(uuid4())
        task_id = context.task_id or str(uuid4())
        metadata = context.message.metadata or {} if context.message else {}
        trace_id = metadata.get("trace_id", str(uuid4()))
        depth = int(metadata.get("delegation_depth", 0))

        logger.info(
            "RAGRetrievalAgent executing | task=%s context=%s trace=%s depth=%d",
            task_id, context_id, trace_id, depth,
        )

        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.submit()
        await updater.start_work()

        try:
            # Call the retrieve pipeline from task 9
            logger.info("Executing retrieval pipeline for query: %s", question)
            results = retrieve(question, top_k=5)
            
            # Format results as a clean JSON list of dicts
            formatted_results = json.dumps(results, ensure_ascii=False, indent=2)

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=formatted_results))],
                name="retrieval_context",
            )
            await updater.complete()

        except Exception as exc:
            logger.exception("RAGRetrievalAgent execution error: %s", exc)
            await updater.failed(
                updater.new_agent_message(
                    parts=[Part(root=TextPart(text=f"Retrieval failed: {exc}"))]
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid4())
        context_id = context.context_id or str(uuid4())
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()

    @staticmethod
    def _extract_question(context: RequestContext) -> str:
        if context.message and context.message.parts:
            parts = []
            for part in context.message.parts:
                inner = getattr(part, "root", part)
                text = getattr(inner, "text", None)
                if text:
                    parts.append(text)
            return "\n".join(parts)
        return ""
