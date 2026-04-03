# -*- coding: utf-8 -*-
"""Minimal action router for runtime orchestration."""

from __future__ import annotations

from typing import Any, Dict

from agent_runtime.tools.explain_op import explain_op_tool
from agent_runtime.tools.generate_yaml import generate_yaml_tool
from agent_runtime.tools.list_ops import list_ops_tool
from agent_runtime.tools.optimize_yaml import optimize_yaml_tool
from agent_runtime.tools.validate_yaml import validate_yaml_tool


class Router:
    """Rule-based placeholder router for future actions."""

    def route(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Route actions to tool wrappers and return normalized envelopes."""
        safe_payload = dict(payload)

        if action == "generate":
            return generate_yaml_tool(
                intent=str(safe_payload.get("intent", "")),
                dataset_path=str(safe_payload.get("dataset_path", "")),
                model_config_path=str(safe_payload.get("model_config_path", "")),
                options=dict(safe_payload.get("options", {})),
            )

        if action == "optimize":
            return optimize_yaml_tool(
                yaml_text_or_path=str(safe_payload.get("yaml_text_or_path", "")),
                objective=str(safe_payload.get("objective", "")),
                model_config_path=str(safe_payload.get("model_config_path", "")),
                options=dict(safe_payload.get("options", {})),
            )

        if action == "list_ops":
            return list_ops_tool(options=dict(safe_payload.get("options", {})))

        if action == "explain_op":
            return explain_op_tool(
                operator_name=str(safe_payload.get("operator_name", "")),
                options=dict(safe_payload.get("options", {})),
            )

        if action == "validate":
            return validate_yaml_tool(
                yaml_text_or_path=str(safe_payload.get("yaml_text_or_path", "")),
                options=dict(safe_payload.get("options", {})),
            )

        return {
            "ok": False,
            "data": {},
            "timing_ms": 1,
            "error": f"unsupported action: {action}",
        }
