# -*- coding: utf-8 -*-
"""Operator discovery screen logic for list/explain actions."""

from __future__ import annotations

from typing import Optional

from apps.tui.runtime_boundary import (
    RuntimeResponse,
    RuntimeServiceLike,
    create_runtime_service,
    dispatch_action,
)


class OperatorScreen:
    """Operator route handlers using runtime service boundary."""

    def __init__(self, service: Optional[RuntimeServiceLike] = None) -> None:
        self.service: RuntimeServiceLike = service or create_runtime_service()

    def list_operators(self) -> RuntimeResponse:
        """Return operator catalog from runtime service."""
        return dispatch_action(self.service, action="list", payload={"options": {"route": "operator"}})

    def explain_operator(self, operator_name: str) -> RuntimeResponse:
        """Return details for one operator."""
        return dispatch_action(
            self.service,
            action="explain",
            payload={"operator_name": operator_name, "options": {"route": "operator"}},
        )
