# -*- coding: utf-8 -*-
"""Minimal YAML validation tool wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from agentic_planner.contracts.recipe import validate_executable_config


def validate_yaml_tool(yaml_text_or_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Validate YAML structure and return normalized envelope."""
    if not yaml_text_or_path:
        return {"ok": False, "data": {}, "timing_ms": 1, "error": "missing required argument: yaml_text_or_path"}

    path = Path(yaml_text_or_path)
    yaml_text = path.read_text(encoding="utf-8") if path.is_file() else yaml_text_or_path
    try:
        parsed = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return {
            "ok": False,
            "data": {"valid": False, "errors": [str(exc)], "options": dict(options)},
            "timing_ms": 1,
            "error": "invalid yaml",
        }

    errors = validate_executable_config(parsed)
    error_message = "validation failed" if errors else None
    return {
        "ok": not errors,
        "data": {"valid": not errors, "errors": errors, "options": dict(options)},
        "timing_ms": 1,
        "error": error_message,
    }
