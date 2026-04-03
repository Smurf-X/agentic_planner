# -*- coding: utf-8 -*-
"""Workflow form and route behavior tests for TUI MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from agent_runtime.api.service import AgentRuntimeService
from apps.tui.screens.workflow_screen import WorkflowFormData, WorkflowScreen


@dataclass
class FakeResponse:
    """Minimal response envelope used by fake service."""

    ok: bool
    data: Dict[str, Any]
    error: str = ""


class FakeService:
    """Deterministic fake runtime service for workflow tests."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def dispatch(self, action: str, payload: Dict[str, Any]) -> FakeResponse:
        self.calls.append({"action": action, "payload": payload})
        if action == "generate":
            return FakeResponse(ok=True, data={"yaml_text": "process:\n  - clean_text_mapper: {}\n"})
        return FakeResponse(ok=True, data={"optimized_yaml": payload["yaml_text_or_path"]})


def test_workflow_form_enforces_required_fields() -> None:
    """Workflow form must include all required inputs."""
    missing = WorkflowFormData(
        task_description="",
        dataset_path="",
        optimization_preference="",
        model_config_path="",
    ).missing_required_fields()

    assert missing == [
        "task_description",
        "dataset_path",
        "optimization_preference",
        "model_config_path",
    ]


def test_workflow_screen_routes_generate_and_optimize() -> None:
    """Workflow screen should route generate then optimize via runtime service."""
    service = FakeService()
    screen = WorkflowScreen(service=service)
    form = WorkflowFormData(
        task_description="clean noisy text",
        dataset_path="/tmp/data.jsonl",
        optimization_preference="quality",
        model_config_path="/tmp/models.yaml",
    )

    generate_result = screen.submit_generate(form)
    optimize_result = screen.submit_optimize(form, yaml_text=generate_result.data["yaml_text"])

    assert generate_result.ok is True
    assert optimize_result.ok is True
    assert [call["action"] for call in service.calls] == ["generate", "optimize"]


def test_workflow_screen_real_service_supports_dispatch() -> None:
    """Workflow screen should use real runtime dispatch without fallback boundary errors."""
    screen = WorkflowScreen(service=AgentRuntimeService())

    result = screen.validate_yaml("process:\n  - clean_text_mapper: {}\n")

    assert result.error != "service does not support dispatch"
    assert isinstance(result.ok, bool)
