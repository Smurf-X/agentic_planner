# -*- coding: utf-8 -*-
"""Tests for agent runtime service contracts."""

from agent_runtime.api.service import AgentRuntimeService
from agent_runtime.orchestrator.session_state import SessionState


def test_service_create_session_returns_id() -> None:
    """Service should create a session and return a non-empty id."""
    service = AgentRuntimeService()
    resp = service.create_session()

    assert resp.ok is True
    assert resp.data["session_id"]


def test_session_state_defaults_and_types() -> None:
    """Session state should initialize contract fields with expected defaults."""
    state = SessionState(session_id="s1")

    assert state.session_id == "s1"
    assert state.current_yaml == ""
    assert state.last_generated_yaml == ""
    assert isinstance(state.last_optimized_candidates, list)
    assert state.last_optimized_candidates == []
    assert state.objective is None
    assert state.dataset_path is None
    assert state.model_config_path is None
    assert isinstance(state.event_history, list)
    assert state.event_history == []


def test_session_state_mutable_defaults_are_isolated() -> None:
    """Session state list defaults should not leak across instances."""
    first = SessionState(session_id="s1")
    second = SessionState(session_id="s2")

    first.last_optimized_candidates.append({"score": 0.9})
    first.event_history.append({"event": "created"})

    assert second.last_optimized_candidates == []
    assert second.event_history == []
    assert first.last_optimized_candidates is not second.last_optimized_candidates
    assert first.event_history is not second.event_history
