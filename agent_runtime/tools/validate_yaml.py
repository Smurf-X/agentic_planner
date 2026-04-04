# -*- coding: utf-8 -*-
"""Minimal YAML validation tool wrapper."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import yaml  # type: ignore[import-untyped]

from agentic_planner.contracts.recipe import validate_executable_config
from agent_runtime.api.schemas import ToolResponse
from agent_runtime.tools.envelope import error_response, ok_response
from agent_runtime.tools.yaml_input import resolve_yaml_input


def validate_yaml_tool(
    yaml_text_or_path: str, options: Optional[Mapping[str, Any]]
) -> ToolResponse:
    """Validate YAML structure and return normalized envelope."""
    if not yaml_text_or_path:
        return error_response("missing required argument: yaml_text_or_path")

    safe_options = dict(options) if isinstance(options, Mapping) else {}

    yaml_input = resolve_yaml_input(yaml_text_or_path=yaml_text_or_path)
    if not yaml_input.ok:
        return yaml_input

    yaml_text = str(yaml_input.data["yaml_text"])
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return error_response(
            "invalid yaml",
            data={"valid": False, "errors": [str(exc)], "options": safe_options},
        )

    errors = validate_executable_config(parsed)
    error_message = "validation failed" if errors else None
    if error_message:
        return error_response(
            error_message,
            data={"valid": False, "errors": errors, "options": safe_options},
        )
    return ok_response(data={"valid": True, "errors": [], "options": safe_options})
