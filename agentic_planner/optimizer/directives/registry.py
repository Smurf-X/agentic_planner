# -*- coding: utf-8 -*-
"""Registry for optimization directives."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentic_planner.optimizer.directives.base import Directive
from agentic_planner.optimizer.directives.adjust_params import BumpMinLenDirective
from agentic_planner.optimizer.directives.adjust_threshold import (
    AdjustThresholdDirective,
    LoosenFiltersDirective,
    TightenFiltersDirective,
)
from agentic_planner.optimizer.directives.remove_redundant import RemoveRedundantOpDirective
from agentic_planner.optimizer.directives.reorder import ReorderFiltersFirstDirective
from agentic_planner.optimizer.op_locator import OpLocator

# Core registry with singleton instances
DIRECTIVE_REGISTRY: Dict[str, Directive] = {}


def _register_instance(d: Directive) -> None:
    """Register a directive instance under its name."""
    DIRECTIVE_REGISTRY[d.name] = d


# Register default directives
_register_instance(ReorderFiltersFirstDirective())
_register_instance(RemoveRedundantOpDirective())
_register_instance(TightenFiltersDirective())
_register_instance(LoosenFiltersDirective())
_register_instance(BumpMinLenDirective(10))


def register_directive(directive: Directive, name: Optional[str] = None) -> str:
    """
    Register a custom directive instance.

    Args:
        directive: The directive instance to register
        name: Optional custom name (default: use directive.name)

    Returns:
        The name under which the directive was registered
    """
    key = name or directive.name
    DIRECTIVE_REGISTRY[key] = directive
    return key


def get_directive(name: str) -> Optional[Directive]:
    """Get a directive by name."""
    return DIRECTIVE_REGISTRY.get(name)


def list_directive_names() -> List[str]:
    """List all registered directive names."""
    return sorted(DIRECTIVE_REGISTRY.keys())


def clear_dynamic_directives() -> None:
    """
    Clear dynamically registered directives (keep core ones).

    Useful for testing or resetting state.
    """
    core_names = {
        "reorder_filters_first",
        "remove_redundant_ops",
        "tighten_filters",
        "loosen_filters",
        "bump_text_length_min_len",
    }
    to_remove = [k for k in DIRECTIVE_REGISTRY if k not in core_names]
    for k in to_remove:
        del DIRECTIVE_REGISTRY[k]


# Convenience registration functions

def register_threshold_directive(
    op_type: str,
    param_name: str,
    delta: float,
    direction: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """Register a parameterized threshold adjustment directive."""
    d = AdjustThresholdDirective(op_type, param_name, delta, direction)
    key = name or f"adjust_{op_type}_{param_name}"
    DIRECTIVE_REGISTRY[key] = d
    return key