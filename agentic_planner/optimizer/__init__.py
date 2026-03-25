# -*- coding: utf-8 -*-
"""
Optimizer module - Pipeline optimization for Data-Juicer.

This module provides:
- Directive-based optimization (Stage 1)
- Search-based optimization (Stage 2)
- Evaluation with LLM-as-a-judge
- Cost tracking and quality scoring
- Stable operator location via OpLocator
"""

# Re-export op_locator for convenience
from agentic_planner.optimizer.op_locator import (
    OpIdentity,
    OpLocator,
    ProcessIndex,
)

__all__ = [
    # Operator location
    "OpIdentity",
    "OpLocator",
    "ProcessIndex",
    # Directive engine (lazy import)
    # "DirectiveEngine",
    # "DirectiveEngineConfig",
    # Search strategies (lazy import)
    # "BeamSearchStrategy",
    # "GreedySearchStrategy",
]


def get_directive_engine():
    """Lazy import for directive engine."""
    from agentic_planner.optimizer.directive_engine import DirectiveEngine
    return DirectiveEngine


def get_search_strategies():
    """Lazy import for search strategies."""
    from agentic_planner.optimizer.search import (
        BeamSearchStrategy,
        GreedySearchStrategy,
        RandomSearchStrategy,
    )
    return BeamSearchStrategy, GreedySearchStrategy, RandomSearchStrategy
