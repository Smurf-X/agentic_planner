# -*- coding: utf-8 -*-
"""Smoke tests for TUI bootstrap and top-level routes."""

from apps.tui.main import AgentPlannerTUI


def test_tui_app_imports() -> None:
    """App class should be importable for CLI bootstrap."""
    assert AgentPlannerTUI is not None


def test_tui_exposes_required_menu_routes() -> None:
    """Top-level TUI should expose all MVP routes."""
    app = AgentPlannerTUI()

    assert app.get_menu_routes() == ["generate", "optimize", "operator", "chat", "export"]
