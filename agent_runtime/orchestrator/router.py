# -*- coding: utf-8 -*-
"""Minimal action router for runtime orchestration."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

from agent_runtime.api.schemas import ToolResponse
from agent_runtime.tools.explain_op import explain_op_tool
from agent_runtime.tools.envelope import error_response
from agent_runtime.tools.generate_yaml import generate_yaml_tool
from agent_runtime.tools.list_ops import list_ops_tool
from agent_runtime.tools.optimize_yaml import optimize_yaml_tool
from agent_runtime.tools.validate_yaml import validate_yaml_tool


class Router:
    """Rule-based placeholder router for future actions."""

    @staticmethod
    def _coerce_payload(payload: Any) -> Optional[Dict[str, Any]]:
        """Coerce payload to a dict when it is mapping-like."""
        if isinstance(payload, Mapping):
            return dict(payload)
        return None

    @staticmethod
    def _coerce_options(raw_options: Any) -> Dict[str, Any]:
        """Coerce options to a dictionary; fallback to empty mapping."""
        if isinstance(raw_options, Mapping):
            return dict(raw_options)
        return {}

    @staticmethod
    def _coerce_required_text(payload: Mapping[str, Any], field: str, default: str = "") -> Union[str, ToolResponse]:
        """Normalize required text fields while preserving explicit type errors."""
        raw_value = payload.get(field)
        if raw_value is None:
            return default
        if isinstance(raw_value, str):
            return raw_value
        return error_response(
            f"invalid type for {field}",
            data={
                "field": field,
                "expected_type": "str",
                "actual_type": type(raw_value).__name__,
            },
        )

    def route(self, action: str, payload: Any) -> ToolResponse:
        """Route actions to tool wrappers and return normalized envelopes."""
        safe_payload = self._coerce_payload(payload)
        if safe_payload is None:
            return error_response(
                "invalid payload: expected mapping",
                data={"payload_type": type(payload).__name__},
            )

        safe_options = self._coerce_options(safe_payload.get("options"))

        if action == "generate":
            intent = self._coerce_required_text(safe_payload, "intent")
            if isinstance(intent, ToolResponse):
                return intent
            dataset_path = self._coerce_required_text(safe_payload, "dataset_path")
            if isinstance(dataset_path, ToolResponse):
                return dataset_path
            llm_config_path = self._coerce_required_text(safe_payload, "model_config_path")
            if isinstance(llm_config_path, ToolResponse):
                return llm_config_path
            return generate_yaml_tool(
                intent=intent,
                dataset_path=dataset_path,
                llm_config_path=llm_config_path,
                options=safe_options,
                payload=safe_payload,
            )

        if action == "optimize":
            yaml_text_or_path = self._coerce_required_text(safe_payload, "yaml_text_or_path")
            if isinstance(yaml_text_or_path, ToolResponse):
                return yaml_text_or_path
            objective = self._coerce_required_text(safe_payload, "objective")
            if isinstance(objective, ToolResponse):
                return objective
            llm_config_path = self._coerce_required_text(safe_payload, "model_config_path")
            if isinstance(llm_config_path, ToolResponse):
                return llm_config_path
            return optimize_yaml_tool(
                yaml_text_or_path=yaml_text_or_path,
                objective=objective,
                llm_config_path=llm_config_path,
                options=safe_options,
                payload=safe_payload,
            )

        if action in {"list", "list_ops"}:
            return list_ops_tool(options=safe_options)

        if action in {"explain", "explain_op"}:
            operator_name = self._coerce_required_text(safe_payload, "operator_name")
            if isinstance(operator_name, ToolResponse):
                return operator_name
            return explain_op_tool(
                operator_name=operator_name,
                options=safe_options,
            )

        if action == "validate":
            yaml_text_or_path = self._coerce_required_text(safe_payload, "yaml_text_or_path")
            if isinstance(yaml_text_or_path, ToolResponse):
                return yaml_text_or_path
            return validate_yaml_tool(
                yaml_text_or_path=yaml_text_or_path,
                options=safe_options,
            )

        return error_response(f"unsupported action: {action}")