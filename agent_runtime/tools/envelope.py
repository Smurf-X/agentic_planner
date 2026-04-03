# -*- coding: utf-8 -*-
"""Shared envelope helpers for runtime tool responses."""

from __future__ import annotations

from typing import Any, Dict, Optional

from agent_runtime.api.schemas import ToolResponse


def ok_response(
    data: Dict[str, Any],
    *,
    timing_ms: int = 1,
    token_usage: Optional[Dict[str, Any]] = None,
) -> ToolResponse:
    """Create a normalized success response envelope."""
    return ToolResponse(
        ok=True, data=data, timing_ms=timing_ms, error=None, token_usage=token_usage
    )


def error_response(
    error: str,
    *,
    data: Optional[Dict[str, Any]] = None,
    timing_ms: int = 1,
) -> ToolResponse:
    """Create a normalized error response envelope."""
    return ToolResponse(ok=False, data=data or {}, timing_ms=timing_ms, error=error)
