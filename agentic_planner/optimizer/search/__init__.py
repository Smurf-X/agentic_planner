# -*- coding: utf-8 -*-
"""
Search strategies for pipeline optimization.

This module provides pluggable search strategies:
- Greedy: Hill-climbing, always pick best immediate improvement
- Random: Random sampling of configurations
- MOAR: Monte-Carlo Operator Action Rollouts
"""

from agentic_planner.optimizer.search.base import (
    BaseSearchStrategy,
    OptimizationObjective,
    SearchConfig,
    SearchReport,
    SearchResult,
    SearchStrategy,
    SearchStrategyType,
)
from agentic_planner.optimizer.search.greedy import (
    GreedySearchConfig,
    GreedySearchStrategy,
)
from agentic_planner.optimizer.search.moar import (
    MOARSearchConfig,
    MOARSearchStrategy,
)
from agentic_planner.optimizer.search.random import (
    RandomSearchConfig,
    RandomSearchStrategy,
)

__all__ = [
    # Base types
    "BaseSearchStrategy",
    "OptimizationObjective",
    "SearchConfig",
    "SearchReport",
    "SearchResult",
    "SearchStrategy",
    "SearchStrategyType",
    # Greedy search
    "GreedySearchConfig",
    "GreedySearchStrategy",
    # Random search
    "RandomSearchConfig",
    "RandomSearchStrategy",
    # MOAR search
    "MOARSearchConfig",
    "MOARSearchStrategy",
]


def create_search_strategy(
    strategy_type: str,
    config: dict,
    evaluator=None,
) -> BaseSearchStrategy:
    """
    Factory function to create search strategies.

    Args:
        strategy_type: Type of search ("greedy", "random", "mcts")
        config: Configuration dict for the strategy
        evaluator: Evaluator for scoring configurations

    Returns:
        A search strategy instance
    """
    if strategy_type == "greedy":
        return GreedySearchStrategy(
            GreedySearchConfig.model_validate(config),
            evaluator,
        )
    elif strategy_type == "random":
        return RandomSearchStrategy(
            RandomSearchConfig.model_validate(config),
            evaluator,
        )
    elif strategy_type == "mcts":
        return MOARSearchStrategy(
            MOARSearchConfig.model_validate(config),
            evaluator,
        )
    else:
        raise ValueError(f"Unknown search strategy: {strategy_type}")
