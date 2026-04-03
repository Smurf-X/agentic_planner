# -*- coding: utf-8 -*-
"""Minimal operator explanation tool wrapper."""

from __future__ import annotations

from typing import Any, Dict

from agent_runtime.api.schemas import ToolResponse
from agent_runtime.tools.envelope import error_response, ok_response


def explain_op_tool(operator_name: str, options: Dict[str, Any]) -> ToolResponse:
    """Return deterministic explanation for a named operator."""
    if not operator_name:
        return error_response("missing required argument: operator_name")

    explanation = {
        "name": operator_name,
        "summary": f"Stub explanation for {operator_name}.",
        "params": [{"name": "text_key", "type": "str", "required": False}],
        "options": dict(options),
    }
    return ok_response(data=explanation)
