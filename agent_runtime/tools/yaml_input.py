# -*- coding: utf-8 -*-
"""Shared YAML input resolution helpers for runtime tools."""

from __future__ import annotations

from pathlib import Path

from agent_runtime.api.schemas import ToolResponse
from agent_runtime.tools.envelope import error_response, ok_response


def resolve_yaml_input(yaml_text_or_path: str) -> ToolResponse:
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
