# -*- coding: utf-8 -*-
"""Tests for agent runtime service contracts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_runtime.api.service import AgentRuntimeService
from agent_runtime.orchestrator.session_state import SessionState


def test_service_create_session_returns_id() -> None:
    """Service should create a session and return a non-empty id."""
    service = AgentRuntimeService()
    resp = service.create_session()

    assert resp.ok is True
    assert resp.data["session_id"]


def test_session_state_has_required_fields() -> None:
    """Session state should include the required contract fields."""
    state = SessionState(session_id="s1")

    assert hasattr(state, "current_yaml")
    assert hasattr(state, "last_generated_yaml")
    assert hasattr(state, "last_optimized_candidates")
    assert hasattr(state, "objective")
    assert hasattr(state, "dataset_path")
    assert hasattr(state, "model_config_path")
    assert hasattr(state, "event_history")
