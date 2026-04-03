# -*- coding: utf-8 -*-
"""Deterministic chat routing helpers for TUI MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from apps.tui.runtime_boundary import (
    RuntimeResponse,
    RuntimeServiceLike,
    create_runtime_service,
    dispatch_action,
)


@dataclass
class ChatRoute:
    """Resolved action + payload from one chat message."""

    action: str
    payload: Dict[str, Any]


class ChatScreen:
    """Simple parser and dispatch helpers for chat commands."""

    def __init__(self, service: Optional[RuntimeServiceLike] = None) -> None:
        self.service: RuntimeServiceLike = service or create_runtime_service()

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

    def submit_message(self, message: str) -> RuntimeResponse:
        """Route and dispatch a chat message through runtime service."""
        route = self.route_message(message)
        return dispatch_action(self.service, action=route.action, payload=route.payload)
