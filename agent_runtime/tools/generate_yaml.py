# -*- coding: utf-8 -*-
"""Minimal YAML generation tool wrapper."""

from __future__ import annotations

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


def generate_yaml_tool(
    intent: str,
    dataset_path: str,
    model_config_path: str,
    options: Dict[str, Any],
) -> Dict[str, Any]:
    """Return deterministic generated YAML payload in a normalized envelope."""
    if not intent:
        return _error_envelope("missing required argument: intent")
    if not dataset_path:
        return _error_envelope("missing required argument: dataset_path")
    if not model_config_path:
        return _error_envelope("missing required argument: model_config_path")

    yaml_text = "process:\n  - clean_text_mapper: {}\n"
    data: Dict[str, Any] = {
        "yaml_text": yaml_text,
        "intent": intent,
        "dataset_path": dataset_path,
        "model_config_path": model_config_path,
        "options": dict(options),
    }
    return _ok_envelope(data=data, token_usage=options.get("token_usage"))
