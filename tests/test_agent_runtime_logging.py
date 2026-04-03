# -*- coding: utf-8 -*-
"""Tests for runtime event log emission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agent_runtime.api.service import AgentRuntimeService


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_service_emits_event_log_for_create_session(tmp_path: Path) -> None:
    """Service should append one JSONL event for create_session."""
    service = AgentRuntimeService(log_dir=tmp_path)

    response = service.create_session()

    assert response.ok is True
    log_path = tmp_path / "events.jsonl"
    assert log_path.exists()

    events = _read_jsonl(log_path)
    assert len(events) == 1
    assert events[0]["tool_name"] == "create_session"


def test_service_emits_event_log_for_dispatch(tmp_path: Path) -> None:
    """Service should append JSONL events around routed runtime actions."""
    service = AgentRuntimeService(log_dir=tmp_path)

    response = service.dispatch(
        action="list",
        payload={"options": {}},
    )

    assert response.ok is True
    events = _read_jsonl(tmp_path / "events.jsonl")
    assert len(events) == 1
    assert events[0]["tool_name"] == "dispatch:list"
