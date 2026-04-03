# -*- coding: utf-8 -*-
"""Minimal operator listing tool wrapper."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from agent_runtime.api.schemas import ToolResponse
from agent_runtime.tools.envelope import error_response, ok_response
from agent_runtime.tools.op_search import get_op_searcher_class, list_op_records


def list_ops_tool(options: Optional[Mapping[str, Any]]) -> ToolResponse:
    """Return operator metadata list sourced from Data-Juicer OPSearcher."""
    safe_options = dict(options) if isinstance(options, Mapping) else {}

    if get_op_searcher_class() is None:
        return error_response(
            "op search unavailable: data_juicer not installed",
            data={"operators": [], "options": safe_options},
        )

    records = list_op_records()
    operators: List[Dict[str, Any]] = []
    for record in records:
        operators.append(
            {
                "name": str(getattr(record, "name", "")),
                "category": str(getattr(record, "type", "")),
                "summary": str(getattr(record, "desc", "") or "").strip(),
                "tags": list(getattr(record, "tags", []) or []),
            }
        )

    operators = [operator for operator in operators if operator["name"]]
    operators.sort(key=lambda item: item["name"])
    return ok_response(data={"operators": operators, "options": safe_options})
