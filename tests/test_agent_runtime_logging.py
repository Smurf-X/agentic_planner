# -*- coding: utf-8 -*-
"""Tests for runtime event log emission."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agent_runtime.api.service import AgentRuntimeService
from agent_runtime.telemetry.logger import EventLogger


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


def test_logger_write_failure_does_not_raise(tmp_path: Path, monkeypatch) -> None:
    """Logger write errors should be swallowed as best-effort telemetry."""
    logger = EventLogger(log_dir=tmp_path)

    def _raise_oserror(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise OSError("mock write failure")

    monkeypatch.setattr(Path, "open", _raise_oserror)

    logger.log_event({"tool_name": "dispatch:list"})


def test_service_dispatch_returns_response_when_logger_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Dispatch should still return routed response when telemetry logging fails."""
    service = AgentRuntimeService(log_dir=tmp_path)

    def _raise_oserror(event: Dict[str, Any]) -> None:  # noqa: ARG001
        raise OSError("mock telemetry failure")

    monkeypatch.setattr(service._event_logger, "log_event", _raise_oserror)

    response = service.dispatch(action="list", payload={"options": {}})

    assert response.ok is True
    assert response.error is None
