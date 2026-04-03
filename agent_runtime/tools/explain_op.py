# -*- coding: utf-8 -*-
"""Minimal operator explanation tool wrapper."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from agent_runtime.api.schemas import ToolResponse
from agent_runtime.tools.envelope import error_response, ok_response
from agent_runtime.tools.op_search import get_op_searcher_class, list_op_records


def explain_op_tool(operator_name: str, options: Optional[Mapping[str, Any]]) -> ToolResponse:
    """Return operator explanation sourced from Data-Juicer OPSearcher."""
    if not operator_name:
        return error_response("missing required argument: operator_name")

    safe_options = dict(options) if isinstance(options, Mapping) else {}

    if get_op_searcher_class() is None:
        return error_response(
            "op search unavailable: data_juicer not installed",
            data={"name": operator_name, "options": safe_options},
        )

    records = list_op_records(specified_op_list=[operator_name])
    target_record = None
    for record in records:
        if str(getattr(record, "name", "")) == operator_name:
            target_record = record
            break

    if target_record is None:
        return error_response(
            f"operator not found: {operator_name}",
            data={"name": operator_name, "options": safe_options},
        )

    explanation = {
        "name": operator_name,
        "summary": str(getattr(target_record, "desc", "") or "").strip(),
        "category": str(getattr(target_record, "type", "")),
        "tags": list(getattr(target_record, "tags", []) or []),
        "signature": str(getattr(target_record, "sig", "")),
        "param_desc": str(getattr(target_record, "param_desc", "") or "").strip(),
        "options": safe_options,
    }
    return ok_response(data=explanation)
