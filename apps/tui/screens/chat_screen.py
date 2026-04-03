# -*- coding: utf-8 -*-
"""Deterministic chat routing helpers for TUI MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from agent_runtime.api.service import AgentRuntimeService


@dataclass
class ChatRoute:
    """Resolved action + payload from one chat message."""

    action: str
    payload: Dict[str, Any]


@dataclass
class LocalResponse:
    """Fallback response contract for unsupported service calls."""

    ok: bool
    data: Dict[str, Any]
    error: str = ""


class ChatScreen:
    """Simple parser and dispatch helpers for chat commands."""

    def __init__(self, service: Any = None) -> None:
        self.service = service or AgentRuntimeService()

    def route_message(self, message: str) -> ChatRoute:
        """Map a message to a deterministic runtime action."""
        text = message.strip()
        if text.startswith("/op "):
            return ChatRoute(
                action="explain",
                payload={"operator_name": text[4:].strip(), "options": {"via": "chat"}},
            )
        if text == "/ops":
            return ChatRoute(action="list", payload={"options": {"via": "chat"}})
        if text.startswith("/validate "):
            return ChatRoute(
                action="validate",
                payload={"yaml_text_or_path": text[10:].strip(), "options": {"via": "chat"}},
            )

        return ChatRoute(
            action="generate",
            payload={
                "intent": text,
                "dataset_path": "",
                "model_config_path": "",
                "options": {"via": "chat"},
            },
        )

    def _dispatch(self, action: str, payload: Dict[str, Any]) -> Any:
        if hasattr(self.service, "dispatch"):
            return self.service.dispatch(action=action, payload=payload)
        if hasattr(self.service, "route"):
            return self.service.route(action=action, payload=payload)
        if hasattr(self.service, "router") and hasattr(self.service.router, "route"):
            return self.service.router.route(action=action, payload=payload)
        return LocalResponse(ok=False, data={}, error="service does not support dispatch")

    def submit_message(self, message: str) -> Any:
        """Route and dispatch a chat message through runtime service."""
        route = self.route_message(message)
        return self._dispatch(action=route.action, payload=route.payload)
