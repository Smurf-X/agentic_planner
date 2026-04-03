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
