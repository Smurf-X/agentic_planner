# -*- coding: utf-8 -*-
"""Minimal YAML optimization tool wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def _error_envelope(error: str) -> Dict[str, Any]:
    """Build a normalized failure envelope."""
    return {"ok": False, "data": {}, "timing_ms": 1, "error": error}


def _ok_envelope(data: Dict[str, Any], token_usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a normalized success envelope."""
    response: Dict[str, Any] = {"ok": True, "data": data, "timing_ms": 1}
    if token_usage is not None:
        response["token_usage"] = token_usage
    return response


def optimize_yaml_tool(
    yaml_text_or_path: str,
    objective: str,
    model_config_path: str,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Return deterministic optimization output in a normalized envelope."""
    if not yaml_text_or_path:
        return _error_envelope("missing required argument: yaml_text_or_path")
    if not objective:
        return _error_envelope("missing required argument: objective")
    if not model_config_path:
        return _error_envelope("missing required argument: model_config_path")

    path = Path(yaml_text_or_path)
    input_yaml = path.read_text(encoding="utf-8") if path.is_file() else yaml_text_or_path
    optimized_yaml = input_yaml

    data: Dict[str, Any] = {
        "optimized_yaml": optimized_yaml,
        "objective": objective,
        "model_config_path": model_config_path,
        "options": dict(options),
        "candidate_count": 1,
    }
    return _ok_envelope(data=data, token_usage=options.get("token_usage"))
