# -*- coding: utf-8 -*-
"""
Optimizer module - Pipeline optimization for Data-Juicer.

This module provides:
- Action-based optimization (operator, directive) pairs
- Directive-based optimization (Stage 1)
- Search-based optimization (Stage 2)
- LLM-guided action selection
- Evaluation with LLM-as-a-judge
- Cost tracking and quality scoring
- Stable operator location via OpLocator
- Model registry for multi-model support
"""

from agentic_planner.optimizer.op_locator import (
    OpIdentity,
    OpLocator,
    ProcessIndex,
)
from agentic_planner.optimizer.action import (
    Action,
    ActionSpace,
    ActionSpaceBuilder,
)

__all__ = [
    "OpIdentity",
    "OpLocator",
    "ProcessIndex",
    "Action",
    "ActionSpace",
    "ActionSpaceBuilder",
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


def get_llm_action_selector():
    """Lazy import for LLM action selector."""
    from agentic_planner.optimizer.llm_action_selector import LLMActionSelector

    return LLMActionSelector


def get_model_registry():
    """Lazy import for model registry."""
    from agentic_planner.optimizer.model_registry import (
        ModelRegistry,
        ModelConfig,
        JudgeConfig,
        ModelsConfig,
    )

    return ModelRegistry, ModelConfig, JudgeConfig, ModelsConfig
