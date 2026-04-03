# -*- coding: utf-8 -*-
"""Agent runtime service entrypoints."""

from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

from agent_runtime.api.schemas import ToolResponse
from agent_runtime.orchestrator.router import Router
from agent_runtime.orchestrator.session_state import SessionState


class AgentRuntimeService:
    """Service facade for runtime operations."""

    def __init__(self) -> None:
        """Initialize in-memory session storage."""
        self._sessions: Dict[str, SessionState] = {}
        self.router = Router()

    def create_session(self) -> ToolResponse:
        """Create a new runtime session and return its identifier."""
        session_id = uuid4().hex
        self._sessions[session_id] = SessionState(session_id=session_id)
        return ToolResponse(ok=True, data={"session_id": session_id})

    def dispatch(self, action: str, payload: Dict[str, Any]) -> ToolResponse:
        """Dispatch one action payload through the runtime router."""
        return self.router.route(action=action, payload=payload)

    def route(self, action: str, payload: Dict[str, Any]) -> ToolResponse:
        """Backward-compatible alias for dispatch callers."""
        return self.dispatch(action=action, payload=payload)
