# -*- coding: utf-8 -*-
"""Bridge between FastAPI routes and AgentRuntimeService."""

from __future__ import annotations

from typing import Any, Dict

from agent_runtime.api.schemas import ToolResponse
from agent_runtime.api.service import AgentRuntimeService


class ServiceBridge:
    """Wraps AgentRuntimeService for web context."""

    def __init__(self) -> None:
        self._service = AgentRuntimeService()

    def dispatch(self, action: str, payload: Dict[str, Any]) -> ToolResponse:
        return self._service.dispatch(action=action, payload=payload)
