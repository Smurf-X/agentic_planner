# -*- coding: utf-8 -*-
"""Minimal operator listing tool wrapper."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from agent_runtime.api.schemas import ToolResponse
from agent_runtime.tools.envelope import error_response, ok_response
from agent_runtime.tools.op_search import get_op_searcher_class, list_op_records


def list_ops_tool(options: Optional[Mapping[str, Any]]) -> ToolResponse:
    """Return operator names list (lightweight for UI rendering)."""
    safe_options = dict(options) if isinstance(options, Mapping) else {}

    if get_op_searcher_class() is None:
        return error_response(
            "op search unavailable: data_juicer not installed",
            data={"operators": [], "options": safe_options},
        )

    records = list_op_records()
    operators: List[str] = []
    for record in records:
        name = str(getattr(record, "name", ""))
        if name:
            operators.append(name)

    operators.sort()
    return ok_response(data={"operators": operators, "count": len(operators)})
