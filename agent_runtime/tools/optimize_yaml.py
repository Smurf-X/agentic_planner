# -*- coding: utf-8 -*-
"""Minimal YAML optimization tool wrapper."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from agent_runtime.api.schemas import ToolResponse
from agent_runtime.tools.envelope import error_response, ok_response
from agent_runtime.tools.yaml_input import resolve_yaml_input


def optimize_yaml_tool(
    yaml_text_or_path: str,
    objective: str,
    model_config_path: str,
    options: Optional[Mapping[str, Any]],
) -> ToolResponse:
    """Return deterministic optimization output in a normalized envelope."""
    if not yaml_text_or_path:
        return error_response("missing required argument: yaml_text_or_path")
    if not objective:
        return error_response("missing required argument: objective")
    if not model_config_path:
        return error_response("missing required argument: model_config_path")

    yaml_input = resolve_yaml_input(yaml_text_or_path=yaml_text_or_path)
    if not yaml_input.ok:
        return yaml_input
    optimized_yaml = str(yaml_input.data["yaml_text"])
    safe_options = dict(options) if isinstance(options, Mapping) else {}

    data: Dict[str, Any] = {
        "optimized_yaml": optimized_yaml,
        "objective": objective,
        "model_config_path": model_config_path,
        "options": safe_options,
        "candidate_count": 1,
    }
    return ok_response(data=data, token_usage=safe_options.get("token_usage"))
