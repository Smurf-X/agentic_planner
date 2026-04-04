# -*- coding: utf-8 -*-
"""Tests for session persistence save/load behavior."""

from __future__ import annotations

import json

from agent_runtime.api.service import AgentRuntimeService
from agent_runtime.orchestrator.session_state import SessionState


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


def test_load_session_rejects_path_traversal_session_id(tmp_path) -> None:
    """Load should reject traversal-like session ids."""
    outside_snapshot = tmp_path.parent / "outside.json"
    outside_snapshot.write_text('{"session_id": "outside"}', encoding="utf-8")

    service = AgentRuntimeService(storage_dir=tmp_path)
    loaded = service.load_session("../outside")

    assert loaded.ok is False
    assert loaded.error is not None
    assert "invalid session" in loaded.error


def test_session_state_from_dict_sanitizes_optional_and_list_fields() -> None:
    """Parser should coerce optionals and list-of-dict fields safely."""
    state = SessionState.from_dict(
        {
            "session_id": "abc123",
            "objective": 123,
            "dataset_path": ["/tmp/data.jsonl"],
            "model_config_path": {"path": "model.yml"},
            "event_history": [{"kind": "ok"}, "bad", 1],
            "last_optimized_candidates": "not-a-list",
        }
    )

    assert state.objective is None
    assert state.dataset_path is None
    assert state.model_config_path is None
    assert state.event_history == [{"kind": "ok"}]
    assert state.last_optimized_candidates == []


def test_load_session_returns_error_for_malformed_snapshot_json(tmp_path) -> None:
    """Load should return normalized error for invalid JSON snapshot."""
    session_id = "feedbeeffeedbeeffeedbeeffeedbeef"
    snapshot = tmp_path / f"{session_id}.json"
    snapshot.write_text("{not-json", encoding="utf-8")

    service = AgentRuntimeService(storage_dir=tmp_path)
    loaded = service.load_session(session_id)

    assert loaded.ok is False
    assert loaded.error is not None
    assert "invalid session snapshot" in loaded.error


def test_load_session_returns_error_for_non_mapping_snapshot(tmp_path) -> None:
    """Load should return normalized error for non-object snapshots."""
    session_id = "deadbeefdeadbeefdeadbeefdeadbeef"
    snapshot = tmp_path / f"{session_id}.json"
    snapshot.write_text(json.dumps(["unexpected"]), encoding="utf-8")

    service = AgentRuntimeService(storage_dir=tmp_path)
    loaded = service.load_session(session_id)

    assert loaded.ok is False
    assert loaded.error is not None
    assert "invalid session snapshot" in loaded.error
