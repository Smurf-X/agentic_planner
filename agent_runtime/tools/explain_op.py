# -*- coding: utf-8 -*-
"""Minimal operator explanation tool wrapper."""

from __future__ import annotations

from typing import Any, Dict


def explain_op_tool(operator_name: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Return deterministic explanation for a named operator."""
    if not operator_name:
        return {"ok": False, "data": {}, "timing_ms": 1, "error": "missing required argument: operator_name"}

    explanation = {
        "name": operator_name,
        "summary": f"Stub explanation for {operator_name}.",
        "params": [{"name": "text_key", "type": "str", "required": False}],
        "options": dict(options),
    }
    return {"ok": True, "data": explanation, "timing_ms": 1, "error": None}
