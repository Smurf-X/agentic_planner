# -*- coding: utf-8 -*-
"""Tests for runtime event logging schema and defaults."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from agent_runtime.api.service import AgentRuntimeService


REQUIRED_EVENT_FIELDS = {
    "timestamp",
    "tool_name",
    "input_summary",
    "result_summary",
    "duration_ms",
    "token_usage",
    "error",
}


def _read_last_event(path: Path) -> Dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines
    return json.loads(lines[-1])


def test_service_default_log_path_is_agent_runtime_logs(tmp_path: Path, monkeypatch) -> None:
    """Default logger path should be under .agent_runtime/logs."""
    monkeypatch.chdir(tmp_path)
    service = AgentRuntimeService()

    service.create_session()

    log_path = tmp_path / ".agent_runtime" / "logs" / "events.jsonl"
    assert log_path.exists()


def test_event_contains_required_schema_fields(tmp_path: Path) -> None:
    """Each event should include the required logging schema."""
    service = AgentRuntimeService(log_dir=tmp_path)

    service.dispatch(action="unknown-action", payload={})

    event = _read_last_event(tmp_path / "events.jsonl")
    assert set(event.keys()) == REQUIRED_EVENT_FIELDS
    assert isinstance(event["duration_ms"], int)
    assert event["duration_ms"] >= 0
