# -*- coding: utf-8 -*-
"""Helpers for reading operator metadata via Data-Juicer OPSearcher."""

from __future__ import annotations

from typing import Any, List, Optional


def get_op_searcher_class() -> Optional[Any]:
    """Return Data-Juicer OPSearcher class when available."""
    try:
        from data_juicer.tools.op_search import OPSearcher

        return OPSearcher
    except ImportError:
        return None


def list_op_records(*, specified_op_list: Optional[List[str]] = None) -> List[Any]:
    """Return operator records from OPSearcher or empty list when unavailable."""
    op_searcher_class = get_op_searcher_class()
    if op_searcher_class is None:
        return []

    try:
        searcher = op_searcher_class(specified_op_list=specified_op_list, include_formatter=False)
    except Exception:
        return []
    return list(searcher.op_records)
