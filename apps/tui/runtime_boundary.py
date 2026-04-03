# -*- coding: utf-8 -*-
"""Shared runtime boundary for TUI dispatch interactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Union, cast, runtime_checkable

from agent_runtime.api.service import AgentRuntimeService


@runtime_checkable
class RuntimeResponse(Protocol):
    """Minimal response envelope expected by TUI screens."""

    ok: bool
    data: Dict[str, Any]
    error: str


@dataclass
class BoundaryErrorResponse:
    """Fallback response when service lacks compatible dispatch entrypoints."""

    ok: bool
    data: Dict[str, Any]
    error: str = ""


@runtime_checkable
class RuntimeDispatchService(Protocol):
    """Runtime service shape exposing dispatch."""

    def dispatch(self, action: str, payload: Dict[str, Any]) -> RuntimeResponse:
        """Dispatch one action payload through runtime tooling."""


@runtime_checkable
class RuntimeRouteService(Protocol):
    """Legacy runtime service shape exposing route."""

    def route(self, action: str, payload: Dict[str, Any]) -> RuntimeResponse:
        """Route one action payload through runtime tooling."""


@runtime_checkable
class RuntimeRouter(Protocol):
    """Router shape containing route() used by compatibility wrappers."""

    def route(self, action: str, payload: Dict[str, Any]) -> RuntimeResponse:
        """Route one action payload through runtime tooling."""


@runtime_checkable
class RuntimeRouterContainer(Protocol):
    """Service shape that exposes a router instance."""

    router: RuntimeRouter


RuntimeServiceLike = Union[RuntimeDispatchService, RuntimeRouteService, RuntimeRouterContainer]


def create_runtime_service() -> RuntimeDispatchService:
    """Create the default runtime service implementation for TUI use."""
    return AgentRuntimeService()


def dispatch_action(
    service: RuntimeServiceLike,
    *,
    action: str,
    payload: Dict[str, Any],
) -> RuntimeResponse:
    """Dispatch runtime action through shared boundary compatibility logic."""
    dispatch = getattr(service, "dispatch", None)
    if callable(dispatch):
        return cast(RuntimeResponse, dispatch(action=action, payload=payload))

    route = getattr(service, "route", None)
    if callable(route):
        return cast(RuntimeResponse, route(action=action, payload=payload))

    router = getattr(service, "router", None)
    if router is not None:
        router_route = getattr(router, "route", None)
        if callable(router_route):
            return cast(RuntimeResponse, router_route(action=action, payload=payload))

    return BoundaryErrorResponse(ok=False, data={}, error="service does not support dispatch")
