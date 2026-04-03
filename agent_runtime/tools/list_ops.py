# -*- coding: utf-8 -*-
"""Minimal operator listing tool wrapper."""

from __future__ import annotations

from typing import Any, Dict, List


def list_ops_tool(options: Dict[str, Any]) -> Dict[str, Any]:
    """Return deterministic operator metadata list."""
    operators: List[Dict[str, Any]] = [
        {
            "name": "clean_text_mapper",
            "category": "mapper",
            "summary": "Normalize and clean text fields.",
        },
        {
            "name": "language_id_score_filter",
            "category": "filter",
            "summary": "Filter rows by language confidence score.",
        },
    ]
    return {
        "ok": True,
        "data": {"operators": operators, "options": dict(options)},
        "timing_ms": 1,
        "error": None,
    }
