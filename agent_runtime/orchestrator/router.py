# -*- coding: utf-8 -*-
"""Minimal action router for runtime orchestration."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

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
    def _coerce_text(raw_value: Any) -> str:
        """Coerce values to text while preserving None as empty text."""
        if raw_value is None:
            return ""
        return str(raw_value)

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
            return generate_yaml_tool(
                intent=self._coerce_text(safe_payload.get("intent")),
                dataset_path=self._coerce_text(safe_payload.get("dataset_path")),
                model_config_path=self._coerce_text(safe_payload.get("model_config_path")),
                options=safe_options,
            )

        if action == "optimize":
            return optimize_yaml_tool(
                yaml_text_or_path=self._coerce_text(safe_payload.get("yaml_text_or_path")),
                objective=self._coerce_text(safe_payload.get("objective")),
                model_config_path=self._coerce_text(safe_payload.get("model_config_path")),
                options=safe_options,
            )

        if action in {"list", "list_ops"}:
            return list_ops_tool(options=safe_options)

        if action in {"explain", "explain_op"}:
            return explain_op_tool(
                operator_name=self._coerce_text(safe_payload.get("operator_name")),
                options=safe_options,
            )

        if action == "validate":
            return validate_yaml_tool(
                yaml_text_or_path=self._coerce_text(safe_payload.get("yaml_text_or_path")),
                options=safe_options,
            )

        return error_response(f"unsupported action: {action}")
