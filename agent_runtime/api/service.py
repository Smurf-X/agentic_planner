# -*- coding: utf-8 -*-
"""Agent runtime service entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
from uuid import uuid4

from agent_runtime.api.schemas import ToolResponse
from agent_runtime.orchestrator.persistence import SessionPersistence
from agent_runtime.orchestrator.router import Router
from agent_runtime.orchestrator.session_state import SessionState


class AgentRuntimeService:
    """Service facade for runtime operations."""

    def __init__(self, storage_dir: Optional[Union[Path, str]] = None) -> None:
        """Initialize in-memory session storage."""
        self._sessions: Dict[str, SessionState] = {}
        self._persistence = SessionPersistence(storage_dir=storage_dir)
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

    def save_session(self, session_id: str) -> ToolResponse:
        """Save one in-memory session to local persistence."""
        state = self._sessions.get(session_id)
        if state is None:
            return ToolResponse(ok=False, error=f"session not found: {session_id}")

        path = self._persistence.save(state)
        return ToolResponse(ok=True, data={"session_id": session_id, "path": path})

    def load_session(self, session_id: str) -> ToolResponse:
        """Load one persisted session and restore it into memory."""
        try:
            restored = self._persistence.load(session_id)
        except FileNotFoundError:
            return ToolResponse(ok=False, error=f"session not found: {session_id}")
        except ValueError:
            return ToolResponse(ok=False, error=f"invalid session snapshot: {session_id}")

        self._sessions[session_id] = restored
        return ToolResponse(ok=True, data=restored.to_dict())
