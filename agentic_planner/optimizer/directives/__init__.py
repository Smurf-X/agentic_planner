# -*- coding: utf-8 -*-
"""Optimization directives for pipeline transformation."""

from agentic_planner.optimizer.directives.base import Directive, DirectiveResult
from agentic_planner.optimizer.directives.registry import (
    DIRECTIVE_REGISTRY,
    clear_dynamic_directives,
    get_directive,
    list_directive_names,
    register_directive,
    register_threshold_directive,
)
from agentic_planner.optimizer.directives.reorder import ReorderFiltersFirstDirective
from agentic_planner.optimizer.directives.remove_redundant import RemoveRedundantOpDirective
from agentic_planner.optimizer.directives.adjust_threshold import (
    AdjustThresholdDirective,
    LoosenFiltersDirective,
    TightenFiltersDirective,
)
from agentic_planner.optimizer.directives.adjust_params import BumpMinLenDirective

__all__ = [
    # Base
    "Directive",
    "DirectiveResult",
    # Registry
    "DIRECTIVE_REGISTRY",
    "register_directive",
    "get_directive",
    "list_directive_names",
    "clear_dynamic_directives",
    "register_threshold_directive",
    # Directives
    "ReorderFiltersFirstDirective",
    "RemoveRedundantOpDirective",
    "AdjustThresholdDirective",
    "TightenFiltersDirective",
    "LoosenFiltersDirective",
    "BumpMinLenDirective",
]