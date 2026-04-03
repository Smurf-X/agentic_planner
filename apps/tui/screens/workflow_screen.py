# -*- coding: utf-8 -*-
"""Structured workflow screen logic for generate/optimize flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from agent_runtime.api.service import AgentRuntimeService


@dataclass
class WorkflowFormData:
    """Structured form fields required for workflow execution."""

    task_description: str
    dataset_path: str
    optimization_preference: str
    model_config_path: str

    def missing_required_fields(self) -> List[str]:
        """Return required field names that are empty."""
        missing: List[str] = []
        if not self.task_description.strip():
            missing.append("task_description")
        if not self.dataset_path.strip():
            missing.append("dataset_path")
        if not self.optimization_preference.strip():
            missing.append("optimization_preference")
        if not self.model_config_path.strip():
            missing.append("model_config_path")
        return missing


@dataclass
class LocalResponse:
    """Fallback response contract for unsupported service calls."""

    ok: bool
    data: Dict[str, Any]
    error: str = ""


class WorkflowScreen:
    """Workflow handlers that call runtime service through one boundary."""

    def __init__(self, service: Any = None) -> None:
        self.service = service or AgentRuntimeService()

    def _dispatch(self, action: str, payload: Dict[str, Any]) -> Any:
        """Dispatch action via runtime service with compatibility fallbacks."""
        if hasattr(self.service, "dispatch"):
            return self.service.dispatch(action=action, payload=payload)
        if hasattr(self.service, "route"):
            return self.service.route(action=action, payload=payload)
        if hasattr(self.service, "router") and hasattr(self.service.router, "route"):
            return self.service.router.route(action=action, payload=payload)
        return LocalResponse(ok=False, data={}, error="service does not support dispatch")

    def submit_generate(self, form: WorkflowFormData) -> Any:
        """Run generate action using structured workflow form inputs."""
        payload = {
            "intent": form.task_description,
            "dataset_path": form.dataset_path,
            "model_config_path": form.model_config_path,
            "options": {"route": "generate", "objective": form.optimization_preference},
        }
        return self._dispatch(action="generate", payload=payload)

    def submit_optimize(self, form: WorkflowFormData, yaml_text: str) -> Any:
        """Run optimize action using workflow objective and generated YAML."""
        payload = {
            "yaml_text_or_path": yaml_text,
            "objective": form.optimization_preference,
            "model_config_path": form.model_config_path,
            "options": {"route": "optimize", "dataset_path": form.dataset_path},
        }
        return self._dispatch(action="optimize", payload=payload)

    def validate_yaml(self, yaml_text_or_path: str) -> Any:
        """Run validate action for intermediate YAML checks."""
        return self._dispatch(
            action="validate",
            payload={"yaml_text_or_path": yaml_text_or_path, "options": {"route": "validate"}},
        )
