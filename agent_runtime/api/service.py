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
from agent_runtime.telemetry.logger import EventLogger


class AgentRuntimeService:
    """Service facade for runtime operations."""

    def __init__(
        self,
        storage_dir: Optional[Union[Path, str]] = None,
        log_dir: Optional[Union[Path, str]] = None,
    ) -> None:
        """Initialize in-memory session storage."""
        self._sessions: Dict[str, SessionState] = {}
        self._persistence = SessionPersistence(storage_dir=storage_dir)
        self._event_logger = EventLogger(log_dir=log_dir)
        self.router = Router()

    @staticmethod
    def _summarize_payload(payload: Any) -> Dict[str, Any]:
        """Create deterministic payload summary for telemetry."""
        if not isinstance(payload, dict):
            return {"payload_type": type(payload).__name__}
        return {
            "keys": sorted(payload.keys()),
            "size": len(payload),
        }

    @staticmethod
    def _summarize_response(response: ToolResponse) -> Dict[str, Any]:
        """Create deterministic response summary for telemetry."""
        return {
            "ok": response.ok,
            "data_keys": sorted(response.data.keys()),
        }

    def _log_runtime_event(
        self,
        *,
        tool_name: str,
        input_summary: Dict[str, Any],
        response: ToolResponse,
        started_at: float,
    ) -> None:
        """Append one runtime event for a service action."""
        self._event_logger.log_event(
            {
                "timestamp": self._event_logger.now_timestamp(),
                "tool_name": tool_name,
                "input_summary": input_summary,
                "result_summary": self._summarize_response(response),
                "duration_ms": self._event_logger.duration_ms(started_at),
                "token_usage": response.token_usage,
                "error": response.error,
            }
        )

    def create_session(self) -> ToolResponse:
        """Create a new runtime session and return its identifier."""
        started_at = self._event_logger.now_counter()
        session_id = uuid4().hex
        self._sessions[session_id] = SessionState(session_id=session_id)
        response = ToolResponse(ok=True, data={"session_id": session_id})
        self._log_runtime_event(
            tool_name="create_session",
            input_summary={},
            response=response,
            started_at=started_at,
        )
        return response

    def dispatch(self, action: str, payload: Dict[str, Any]) -> ToolResponse:
        """Dispatch one action payload through the runtime router."""
        started_at = self._event_logger.now_counter()
        response = self.router.route(action=action, payload=payload)
        self._log_runtime_event(
            tool_name=f"dispatch:{action}",
            input_summary=self._summarize_payload(payload),
            response=response,
            started_at=started_at,
        )
        return response

    def route(self, action: str, payload: Dict[str, Any]) -> ToolResponse:
        """Backward-compatible alias for dispatch callers."""
        return self.dispatch(action=action, payload=payload)

    def save_session(self, session_id: str) -> ToolResponse:
        """Save one in-memory session to local persistence."""
        started_at = self._event_logger.now_counter()
        state = self._sessions.get(session_id)
        if state is None:
            response = ToolResponse(ok=False, error=f"session not found: {session_id}")
            self._log_runtime_event(
                tool_name="save_session",
                input_summary={"session_id": session_id},
                response=response,
                started_at=started_at,
            )
            return response

        path = self._persistence.save(state)
        response = ToolResponse(ok=True, data={"session_id": session_id, "path": path})
        self._log_runtime_event(
            tool_name="save_session",
            input_summary={"session_id": session_id},
            response=response,
            started_at=started_at,
        )
        return response

    def load_session(self, session_id: str) -> ToolResponse:
        """Load one persisted session and restore it into memory."""
        started_at = self._event_logger.now_counter()
        try:
            restored = self._persistence.load(session_id)
        except FileNotFoundError:
            response = ToolResponse(ok=False, error=f"session not found: {session_id}")
            self._log_runtime_event(
                tool_name="load_session",
                input_summary={"session_id": session_id},
                response=response,
                started_at=started_at,
            )
            return response
        except ValueError:
            response = ToolResponse(ok=False, error=f"invalid session snapshot: {session_id}")
            self._log_runtime_event(
                tool_name="load_session",
                input_summary={"session_id": session_id},
                response=response,
                started_at=started_at,
            )
            return response

        self._sessions[session_id] = restored
        response = ToolResponse(ok=True, data=restored.to_dict())
        self._log_runtime_event(
            tool_name="load_session",
            input_summary={"session_id": session_id},
            response=response,
            started_at=started_at,
        )
        return response
