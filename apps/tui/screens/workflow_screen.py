# -*- coding: utf-8 -*-
"""Structured workflow screen logic for generate/optimize flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from apps.tui.runtime_boundary import (
    RuntimeResponse,
    RuntimeServiceLike,
    create_runtime_service,
    dispatch_action,
)


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


class WorkflowScreen:
    """Workflow handlers that call runtime service through one boundary."""

    def __init__(self, service: Optional[RuntimeServiceLike] = None) -> None:
        self.service: RuntimeServiceLike = service or create_runtime_service()

    def submit_generate(self, form: WorkflowFormData) -> RuntimeResponse:
        """Run generate action using structured workflow form inputs."""
        payload = {
            "intent": form.task_description,
            "dataset_path": form.dataset_path,
            "model_config_path": form.model_config_path,
            "options": {"route": "generate", "objective": form.optimization_preference},
        }
        return dispatch_action(self.service, action="generate", payload=payload)

    def submit_optimize(self, form: WorkflowFormData, yaml_text: str) -> RuntimeResponse:
        """Run optimize action using workflow objective and generated YAML."""
        payload = {
            "yaml_text_or_path": yaml_text,
            "objective": form.optimization_preference,
            "model_config_path": form.model_config_path,
            "options": {"route": "optimize", "dataset_path": form.dataset_path},
        }
        return dispatch_action(self.service, action="optimize", payload=payload)

    def validate_yaml(self, yaml_text_or_path: str) -> RuntimeResponse:
        """Run validate action for intermediate YAML checks."""
        return dispatch_action(
            self.service,
            action="validate",
            payload={"yaml_text_or_path": yaml_text_or_path, "options": {"route": "validate"}},
        )
