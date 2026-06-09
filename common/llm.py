"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at Nvidia NIM or OpenRouter."""
    nvidia_key = os.getenv("nvida_key")
    if nvidia_key and nvidia_key.strip():
        return ChatOpenAI(
            model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"),
            openai_api_key=nvidia_key.strip(),
            openai_api_base="https://integrate.api.nvidia.com/v1",
        )
    # Check both cases for OpenRouter API key
    openrouter_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENrouter_API_KEY")
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o"),
        openai_api_key=openrouter_key,
        openai_api_base="https://openrouter.ai/api/v1",
    )