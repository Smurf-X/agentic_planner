# -*- coding: utf-8 -*-
"""Minimal YAML generation tool wrapper."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from agent_runtime.api.schemas import ToolResponse
from agent_runtime.tools.envelope import error_response, ok_response


def generate_yaml_tool(
    intent: str,
    dataset_path: str,
    model_config_path: str,
    options: Optional[Mapping[str, Any]],
) -> ToolResponse:
    """Return deterministic generated YAML payload in a normalized envelope."""
    if not intent:
        return error_response("missing required argument: intent")
    if not dataset_path:
        return error_response("missing required argument: dataset_path")
    if not model_config_path:
        return error_response("missing required argument: model_config_path")

    safe_options = dict(options) if isinstance(options, Mapping) else {}
    yaml_text = "process:\n  - clean_text_mapper: {}\n"
    data: Dict[str, Any] = {
        "yaml_text": yaml_text,
        "intent": intent,
        "dataset_path": dataset_path,
        "model_config_path": model_config_path,
        "options": safe_options,
    }
    return ok_response(data=data, token_usage=safe_options.get("token_usage"))
