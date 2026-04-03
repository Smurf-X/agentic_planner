# -*- coding: utf-8 -*-
"""Lightweight TUI shell with deterministic route wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from apps.tui.runtime_boundary import RuntimeServiceLike, create_runtime_service

from apps.tui.screens.chat_screen import ChatScreen
from apps.tui.screens.export_screen import ExportScreen
from apps.tui.screens.operator_screen import OperatorScreen
from apps.tui.screens.workflow_screen import WorkflowScreen


@dataclass
class AppRoute:
    """Declarative route mapping for the TUI shell."""

    name: str
    screen: Any


class AgentPlannerTUI:
    """Minimal TUI shell that exposes MVP menu routes."""

    def __init__(self, service: Optional[RuntimeServiceLike] = None) -> None:
        self.service: RuntimeServiceLike = service or create_runtime_service()
        self._routes: Dict[str, AppRoute] = {
            "generate": AppRoute(name="generate", screen=WorkflowScreen),
            "optimize": AppRoute(name="optimize", screen=WorkflowScreen),
            "operator": AppRoute(name="operator", screen=OperatorScreen),
            "chat": AppRoute(name="chat", screen=ChatScreen),
            "export": AppRoute(name="export", screen=ExportScreen),
        }

    def get_menu_routes(self) -> List[str]:
        """Return top-level MVP routes in menu order."""
        return ["generate", "optimize", "operator", "chat", "export"]

    def open_route(self, route_name: str) -> Any:
        """Instantiate and return a screen by route name."""
        route = self._routes.get(route_name)
        if route is None:
            raise ValueError(f"unknown route: {route_name}")

        if route.screen in {WorkflowScreen, OperatorScreen}:
            return route.screen(service=self.service)
        if route.screen is ChatScreen:
            return route.screen(service=self.service)
        return route.screen()
