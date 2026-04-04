# -*- coding: utf-8 -*-
"""Tests for deterministic chat intent routing."""

from apps.tui.screens.chat_screen import ChatScreen


def test_chat_routes_generate_message() -> None:
    """Chat route should map plain request to generate action."""
    screen = ChatScreen()

    route = screen.route_message("Generate a cleaning pipeline")

    assert route.action == "generate"
    assert route.payload["intent"] == "Generate a cleaning pipeline"


def test_chat_routes_operator_lookup_command() -> None:
    """Chat route should map /op command to explain action."""
    screen = ChatScreen()

    route = screen.route_message("/op clean_text_mapper")

    assert route.action == "explain"
    assert route.payload["operator_name"] == "clean_text_mapper"


def test_chat_routes_ops_command() -> None:
    """Chat route should map /ops command to list action."""
    screen = ChatScreen()

    route = screen.route_message("/ops")

    assert route.action == "list"
    assert route.payload["options"]["via"] == "chat"


def test_chat_routes_validate_command() -> None:
    """Chat route should map /validate command to validate action."""
    screen = ChatScreen()

    route = screen.route_message("/validate process:\n  - clean_text_mapper: {}\n")

    assert route.action == "validate"
    assert route.payload["yaml_text_or_path"].startswith("process:")
