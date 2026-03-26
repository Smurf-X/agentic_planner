# -*- coding: utf-8 -*-
"""Strict parameter allowlists from operator ``__init__`` signatures (no LLM-invented keys)."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Tuple


def _get_operators_registry() -> Dict[str, Any]:
    """
    Get the operators registry from data_juicer.
    
    This is a lazy import to avoid hard dependency issues.
    """
    try:
        from data_juicer.ops.base_op import OPERATORS
        return OPERATORS.modules
    except ImportError:
        # Fallback: return empty dict if data_juicer not installed
        return {}


def get_operator_class(op_name: str) -> type:
    operators = _get_operators_registry()
    if op_name not in operators:
        raise KeyError(f"unknown operator: {op_name}")
    return operators[op_name]


def get_init_param_allowlist(op_name: str) -> List[str]:
    """
    Names of ``__init__`` parameters for the concrete operator class, excluding ``self``
    and ``*args`` / ``**kwargs`` buckets. These are the only keys allowed under this op in
    ``process`` YAML (plus globals at recipe root such as ``text_keys``).
    """
    cls = get_operator_class(op_name)
    sig = inspect.signature(cls.__init__)
    out: List[str] = []
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        out.append(pname)
    return out


def _format_default(d: Any) -> str:
    if d is inspect.Parameter.empty:
        return "(required)"
    return repr(d)


def format_allowlist_for_prompt(op_name: str) -> str:
    """Compact multi-line block: ``name: type, default=...`` for LLM."""
    cls = get_operator_class(op_name)
    sig = inspect.signature(cls.__init__)
    lines: List[str] = []
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        ann = param.annotation
        ann_s = getattr(ann, "__name__", str(ann)) if ann is not inspect.Parameter.empty else "Any"
        lines.append(f"  - {pname}: {ann_s}, default={_format_default(param.default)}")
    if not lines:
        return "  (no explicit parameters; use empty params {{}})"
    return "\n".join(lines)


def sanitize_params(op_name: str, params: Any) -> Dict[str, Any]:
    """Drop any key not in the operator ``__init__`` allowlist."""
    if not isinstance(params, dict):
        return {}
    allow = set(get_init_param_allowlist(op_name))
    return {k: v for k, v in params.items() if k in allow}


def build_schema_block(operator_names: List[str]) -> str:
    """Human-readable block for batch strict-fill prompts."""
    chunks: List[str] = []
    for name in operator_names:
        allow = get_init_param_allowlist(name)
        allow_csv = ", ".join(allow) if allow else "(none — use {{}})"
        body = format_allowlist_for_prompt(name)
        chunks.append(f"### {name}\nAllowed parameter names: {allow_csv}\n{body}")
    return "\n\n".join(chunks)


def validate_params_bind(op_name: str, params: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Try binding ``params`` to ``__init__`` (excluding ``self``). Returns (ok, message).

    Used to catch missing required arguments after sanitization.
    """
    cls = get_operator_class(op_name)
    sig = inspect.signature(cls.__init__)
    params_only = [
        p
        for n, p in sig.parameters.items()
        if n != "self" and p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if not params_only:
        return True, ""

    new_sig = inspect.Signature(parameters=params_only)
    try:
        new_sig.bind(**params)
    except TypeError as exc:
        return False, str(exc)
    return True, ""