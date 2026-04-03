# -*- coding: utf-8 -*-
"""Operator discovery screen logic for list/explain actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from agent_runtime.api.service import AgentRuntimeService


@dataclass
class LocalResponse:
    """Fallback response contract for unsupported service calls."""

    ok: bool
    data: Dict[str, Any]
    error: str = ""


class OperatorScreen:
    """Operator route handlers using runtime service boundary."""

    def __init__(self, service: Any = None) -> None:
        self.service = service or AgentRuntimeService()

    def _dispatch(self, action: str, payload: Dict[str, Any]) -> Any:
        if hasattr(self.service, "dispatch"):
            return self.service.dispatch(action=action, payload=payload)
        if hasattr(self.service, "route"):
            return self.service.route(action=action, payload=payload)
        if hasattr(self.service, "router") and hasattr(self.service.router, "route"):
            return self.service.router.route(action=action, payload=payload)
        return LocalResponse(ok=False, data={}, error="service does not support dispatch")

    def list_operators(self) -> Any:
        """Return operator catalog from runtime service."""
        return self._dispatch(action="list", payload={"options": {"route": "operator"}})

    def explain_operator(self, operator_name: str) -> Any:
        """Return details for one operator."""
        return self._dispatch(
            action="explain",
            payload={"operator_name": operator_name, "options": {"route": "operator"}},
        )
