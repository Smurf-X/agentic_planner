# -*- coding: utf-8 -*-
"""Build compact operator catalog text from :class:`data_juicer.tools.op_search.OPSearcher`."""

from __future__ import annotations

from typing import Iterable, List, Optional, Set


def _get_op_searcher():
    """Lazy import OPSearcher from data_juicer."""
    try:
        from data_juicer.tools.op_search import OPSearcher
        return OPSearcher
    except ImportError:
        return None


def _compact_desc(desc: str, max_len: int = 320) -> str:
    text = (desc or "").strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def build_operator_catalog_text(
    *,
    include_formatter: bool = False,
    exclude_names: Optional[Set[str]] = None,
    only_names: Optional[Set[str]] = None,
) -> str:
    """
    One line per operator: ``name | type | tags | description``.

    Suitable for LLM prompt injection (full catalog, no vector retrieval).

    :param only_names: If set, only include operators whose names appear in this set.
    """
    OPSearcher = _get_op_searcher()
    if OPSearcher is None:
        return "(data_juicer not installed - operator catalog unavailable)"
    
    searcher = OPSearcher(specified_op_list=None, include_formatter=include_formatter)
    exclude_names = exclude_names or set()
    only = only_names
    lines: List[str] = []
    for rec in searcher.op_records:
        if rec.name in exclude_names:
            continue
        if only is not None and rec.name not in only:
            continue
        tags = ",".join(rec.tags) if rec.tags else ""
        line = f"{rec.name} | {rec.type} | {tags} | {_compact_desc(rec.desc)}"
        lines.append(line)
    lines.sort()
    return "\n".join(lines)


def build_operator_detail_text(
    operator_names: Iterable[str],
    *,
    max_param_chars: int = 4000,
) -> str:
    """
    Parameter documentation for a *selected* subset of operators (step 2 of NL generation).

    Uses ``OPSearcher`` metadata (signature + param descriptions).
    """
    names = [n for n in operator_names if n]
    if not names:
        return ""
    
    OPSearcher = _get_op_searcher()
    if OPSearcher is None:
        return "(data_juicer not installed - operator details unavailable)"
    
    searcher = OPSearcher(specified_op_list=list(dict.fromkeys(names)))
    chunks: List[str] = []
    for rec in searcher.op_records:
        param_blob = rec.param_desc or ""
        if len(param_blob) > max_param_chars:
            param_blob = param_blob[: max_param_chars - 3] + "..."
        sig = str(rec.sig)
        chunk = (
            f"### {rec.name}\n"
            f"type: {rec.type}\n"
            f"tags: {rec.tags}\n"
            f"signature: {sig}\n"
            f"params:\n{param_blob}\n"
        )
        chunks.append(chunk)
    return "\n".join(chunks)