# -*- coding: utf-8 -*-
"""Bridge between FastAPI routes and AgentRuntimeService."""

from __future__ import annotations

import time
from typing import Any, Dict

from agent_runtime.api.schemas import ToolResponse
from agent_runtime.api.service import AgentRuntimeService
from agentic_planner.generator import OpenAICompatibleJsonClient


class ServiceBridge:
    """Wraps AgentRuntimeService for web context."""

    def __init__(self) -> None:
        self._service = AgentRuntimeService()

    def dispatch(self, action: str, payload: Dict[str, Any]) -> ToolResponse:
        return self._service.dispatch(action=action, payload=payload)

    def test_llm(self, base_url: str, api_key: str, model: str) -> ToolResponse:
        """Test LLM connection by making a simple API call."""
        if not model:
            return ToolResponse(ok=False, error="Model name is required")
        if not api_key:
            return ToolResponse(ok=False, error="API key is required")

        start_time = time.time()
        try:
            client = OpenAICompatibleJsonClient(
                model=model,
                api_key=api_key,
                base_url=base_url or "https://api.openai.com/v1",
            )
            result = client.generate(
                system_prompt="You are a helpful assistant.",
                user_prompt="Reply with 'ok' only.",
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            client.close()

            return ToolResponse(
                ok=True,
                data={
                    "message": "LLM connection successful",
                    "model": model,
                    "response_preview": result[:100] if result else "",
                },
                timing_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return ToolResponse(
                ok=False,
                error=f"LLM connection failed: {str(e)}",
                timing_ms=elapsed_ms,
            )
