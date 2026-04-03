# -*- coding: utf-8 -*-
"""Agent runtime service entrypoints."""

from typing import Dict
from uuid import uuid4

from agent_runtime.api.schemas import ToolResponse
from agent_runtime.orchestrator.session_state import SessionState


class AgentRuntimeService:
    """Service facade for runtime operations."""

    def __init__(self) -> None:
        """Initialize in-memory session storage."""
        self._sessions: Dict[str, SessionState] = {}

    def create_session(self) -> ToolResponse:
        """Create a new runtime session and return its identifier."""
        session_id = uuid4().hex
        self._sessions[session_id] = SessionState(session_id=session_id)
        return ToolResponse(ok=True, data={"session_id": session_id})
