# -*- coding: utf-8 -*-
"""Tests for session persistence save/load behavior."""

from __future__ import annotations

from agent_runtime.api.service import AgentRuntimeService


def test_session_can_save_and_restore(tmp_path) -> None:
    """Service should persist a session and load it back by id."""
    service = AgentRuntimeService(storage_dir=tmp_path)
    created = service.create_session()
    session_id = created.data["session_id"]

    service.save_session(session_id)
    restored = service.load_session(session_id)

    assert restored.ok is True
    assert restored.data["session_id"] == session_id


def test_load_session_restores_saved_fields(tmp_path) -> None:
    """Saved session fields should survive a fresh service instance."""
    service = AgentRuntimeService(storage_dir=tmp_path)
    session_id = service.create_session().data["session_id"]

    service._sessions[session_id].current_yaml = "process:\n  - clean_text_mapper: {}\n"
    service._sessions[session_id].dataset_path = "/tmp/data.jsonl"
    service.save_session(session_id)

    reloaded_service = AgentRuntimeService(storage_dir=tmp_path)
    restored = reloaded_service.load_session(session_id)

    assert restored.ok is True
    assert restored.data["current_yaml"] == "process:\n  - clean_text_mapper: {}\n"
    assert restored.data["dataset_path"] == "/tmp/data.jsonl"
