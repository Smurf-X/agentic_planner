# -*- coding: utf-8 -*-
"""Typed API response schemas for runtime tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolResponse:
    """Standard tool response envelope."""

    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
