# -*- coding: utf-8 -*-
"""Tests for export preview flow with persisted sessions."""

from __future__ import annotations

from agent_runtime.api.service import AgentRuntimeService
from apps.tui.screens.export_screen import ExportScreen


def test_export_preview_uses_restored_yaml(tmp_path) -> None:
    """Preview should report changes when restored YAML differs."""
    service = AgentRuntimeService(storage_dir=tmp_path)
    session_id = service.create_session().data["session_id"]
    service._sessions[session_id].current_yaml = "process:\n  - clean_text_mapper: {}\n"
    service.save_session(session_id)

    restored = AgentRuntimeService(storage_dir=tmp_path).load_session(session_id)
    screen = ExportScreen()

    preview = screen.preview_before_overwrite(
        existing_yaml=restored.data["current_yaml"],
        next_yaml="process:\n  - clean_text_mapper: {}\n  - remove_special_chars_mapper: {}\n",
    )

    assert preview.changed is True
    assert "Preview before overwrite" in preview.message


def test_export_copy_returns_yaml_unchanged() -> None:
    """Copy helper should preserve YAML text exactly."""
    screen = ExportScreen()
    yaml_text = "process:\n  - clean_text_mapper: {}\n"

    assert screen.copy_yaml_once(yaml_text) == yaml_text
