# -*- coding: utf-8 -*-
"""Smoke tests for TUI bootstrap and top-level routes."""

import pytest

from apps.tui.main import AgentPlannerTUI
from apps.tui.screens.chat_screen import ChatScreen
from apps.tui.screens.export_screen import ExportScreen
from apps.tui.screens.operator_screen import OperatorScreen
from apps.tui.screens.workflow_screen import WorkflowScreen


def test_tui_app_imports() -> None:
    """App class should be importable for CLI bootstrap."""
    assert AgentPlannerTUI is not None


def test_tui_exposes_required_menu_routes() -> None:
    """Top-level TUI should expose all MVP routes."""
    app = AgentPlannerTUI()

    assert app.get_menu_routes() == ["generate", "optimize", "operator", "chat", "export"]


def test_open_route_returns_expected_screen_types() -> None:
    """Route opening should instantiate expected screen classes."""
    app = AgentPlannerTUI()

    assert isinstance(app.open_route("generate"), WorkflowScreen)
    assert isinstance(app.open_route("optimize"), WorkflowScreen)
    assert isinstance(app.open_route("operator"), OperatorScreen)
    assert isinstance(app.open_route("chat"), ChatScreen)
    assert isinstance(app.open_route("export"), ExportScreen)


def test_open_route_raises_for_unknown_route() -> None:
    """Unknown route names should raise a ValueError."""
    app = AgentPlannerTUI()

    with pytest.raises(ValueError, match="unknown route"):
        app.open_route("unknown")
