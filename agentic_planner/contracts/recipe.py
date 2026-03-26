# -*- coding: utf-8 -*-
"""DJ executable config helpers (config_all.yaml style)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Union

import yaml

DJExecutableConfig = Dict[str, Any]


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


def validate_executable_config(cfg: DJExecutableConfig) -> List[str]:
    """
    Lightweight structural validation. Returns a list of error messages (empty if ok).

    Does not execute the pipeline; only checks keys required by the planner contracts.
    """
    errors: List[str] = []
    if not isinstance(cfg, dict):
        return ["config must be a mapping"]

    proc = cfg.get("process")
    if proc is None:
        errors.append("missing required key: process")
        return errors
    if not isinstance(proc, list) or not proc:
        errors.append("process must be a non-empty list")
        return errors

    operators = _get_operators_registry()

    for i, step in enumerate(proc):
        if not isinstance(step, dict) or len(step) != 1:
            errors.append(f"process[{i}] must be a single-key dict {{op_name: params}}")
            continue
        op_name = next(iter(step.keys()))
        params = step[op_name]
        if operators and op_name not in operators:
            errors.append(f"process[{i}]: unknown operator {op_name!r}")
        if not isinstance(params, dict):
            errors.append(f"process[{i}]: params for {op_name!r} must be a dict")

    return errors


def load_executable_config(path: Union[str, Path]) -> DJExecutableConfig:
    """Load a YAML config file into a plain dict."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, MutableMapping):
        raise ValueError(f"config root must be a mapping, got {type(data)}")
    return dict(data)


def save_executable_config(cfg: DJExecutableConfig, path: Union[str, Path]) -> None:
    """Write config to YAML (utf-8)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            cfg,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )