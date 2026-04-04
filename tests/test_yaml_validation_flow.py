# -*- coding: utf-8 -*-
"""Tests for YAML validation and export UX placeholders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from apps.tui.screens.export_screen import ExportScreen
from apps.tui.screens.workflow_screen import WorkflowScreen


@dataclass
class FakeResponse:
    """Minimal response envelope used by fake service."""

    ok: bool
    data: Dict[str, Any]
    error: str = ""


class FakeService:
    """Deterministic fake runtime service for validation flow tests."""

    def dispatch(self, action: str, payload: Dict[str, Any]) -> FakeResponse:
        if action == "validate":
            return FakeResponse(ok=True, data={"valid": True, "errors": []})
        return FakeResponse(ok=False, data={}, error=f"unsupported action: {action}")


def test_workflow_screen_supports_yaml_validation_action() -> None:
    """Workflow should expose YAML validation through runtime dispatch."""
    screen = WorkflowScreen(service=FakeService())

    result = screen.validate_yaml("process:\n  - clean_text_mapper: {}\n")

    assert result.ok is True
    assert result.data["valid"] is True


def test_export_screen_preview_and_copy_placeholders() -> None:
    """Export screen should provide overwrite preview and copy helper."""
    screen = ExportScreen()

    preview = screen.preview_before_overwrite(
        existing_yaml="process: []\n",
        next_yaml="process:\n  - clean_text_mapper: {}\n",
    )
    copied = screen.copy_yaml_once("process:\n  - clean_text_mapper: {}\n")

    assert preview.changed is True
    assert "Preview before overwrite" in preview.message
    assert copied == "process:\n  - clean_text_mapper: {}\n"
