# -*- coding: utf-8 -*-
"""Minimal YAML optimization tool wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from agent_runtime.api.schemas import ToolResponse
from agent_runtime.tools.envelope import error_response, ok_response


def _resolve_yaml_input(yaml_text_or_path: str) -> ToolResponse:
    """Resolve YAML input from raw text or a file path."""
    path = Path(yaml_text_or_path)
    if not path.is_file():
        return ok_response(data={"yaml_text": yaml_text_or_path})

    try:
        yaml_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return error_response(
            "failed to read yaml input",
            data={"reason": str(exc), "yaml_text_or_path": yaml_text_or_path},
        )

    return ok_response(data={"yaml_text": yaml_text})


def optimize_yaml_tool(
    yaml_text_or_path: str,
    objective: str,
    model_config_path: str,
    options: Optional[Dict[str, Any]],
) -> ToolResponse:
    """Return deterministic optimization output in a normalized envelope."""
    if not yaml_text_or_path:
        return error_response("missing required argument: yaml_text_or_path")
    if not objective:
        return error_response("missing required argument: objective")
    if not model_config_path:
        return error_response("missing required argument: model_config_path")

    yaml_input = _resolve_yaml_input(yaml_text_or_path=yaml_text_or_path)
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
