# -*- coding: utf-8 -*-
"""
Search strategy abstractions for pipeline optimization.

This module defines the pluggable search interface that allows
different optimization strategies (greedy, beam search, MCTS, etc.)
to be used interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentic_planner.contracts.cost import CostBreakdown
from agentic_planner.contracts.recipe import DJExecutableConfig


class SearchStrategyType(str, Enum):
    """Available search strategy types."""

    GREEDY = "greedy"
    """Greedy search: always pick the best immediate improvement."""

    RANDOM = "random"
    """Random search: sample configurations randomly."""

    BEAM = "beam"
    """Beam search: keep top-k candidates at each iteration."""

    EVOLUTIONARY = "evolutionary"
    """Evolutionary search: population-based optimization."""

    MCTS = "mcts"
    """Monte Carlo Tree Search: balance exploration/exploitation."""


class OptimizationObjective(str, Enum):
    """What to optimize for."""

    QUALITY = "quality"
    """Maximize quality only."""

    COST = "cost"
    """Minimize cost only."""

    BALANCED = "balanced"
    """Balance quality and cost."""

    PARETO = "pareto"
    """Find Pareto frontier of quality vs cost."""


@dataclass
class SearchResult:
    """Result from a single search step or final result."""

    config: DJExecutableConfig
    cost: CostBreakdown
    quality: float
    origin: str = ""
    trace: List[Any] = field(default_factory=list)
    generation: int = 0
    parent_id: Optional[str] = None


@dataclass
class SearchReport:
    """Complete report from a search optimization run."""

    ok: bool
    candidates: List[SearchResult]
    pareto_front: List[SearchResult] = field(default_factory=list)
    total_iterations: int = 0
    total_evaluations: int = 0
    best_by_quality: Optional[SearchResult] = None
    best_by_cost: Optional[SearchResult] = None
    best_balanced: Optional[SearchResult] = None
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class SearchConfig(BaseModel):
    """Base configuration for search strategies."""

    strategy: SearchStrategyType = Field(
        default=SearchStrategyType.BEAM,
        description="Search strategy to use.",
    )
    objective: OptimizationObjective = Field(
        default=OptimizationObjective.PARETO,
        description="Optimization objective.",
    )
    max_iterations: int = Field(default=10, ge=1, le=100)
    max_evaluations: int = Field(default=100, ge=1, le=1000)
    seed: int = Field(default=42, description="Random seed for reproducibility.")

    model_config = {"extra": "allow"}


@runtime_checkable
class SearchStrategy(Protocol):
    """
    Protocol for search strategies.

    Any search implementation must implement this interface to be
    usable by the optimization runner.
    """

    def search(self, root: DJExecutableConfig) -> SearchReport:
        """
        Execute the search starting from the root configuration.

        Args:
            root: Starting configuration

        Returns:
            SearchReport with all discovered candidates
        """


class BaseSearchStrategy(ABC):
    """
    Base class for search strategies with common utilities.

    Provides shared functionality for:
    - Configuration validation
    - Candidate tracking
    - Pareto frontier computation
    """

    def __init__(
        self,
        config: SearchConfig,
        evaluator: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._evaluator = evaluator
        self._evaluated_count = 0
        self._iteration_count = 0

    @abstractmethod
    def search(self, root: DJExecutableConfig) -> SearchReport:
        """Execute the search strategy."""

    def _evaluate(self, config: DJExecutableConfig) -> tuple[CostBreakdown, float]:
        """Evaluate a configuration and track count."""
        if self._evaluator is None:
            # Return dummy values
            return CostBreakdown(), 0.5
        self._evaluated_count += 1
        return self._evaluator.evaluate(config)

    def _compute_pareto_front(self, candidates: List[SearchResult]) -> List[SearchResult]:
        """
        Compute Pareto frontier for quality vs cost.

        A candidate dominates another if it has:
        - Higher quality AND lower or equal cost, OR
        - Equal quality AND lower cost
        """
        if not candidates:
            return []

        # Sort by quality descending, then by cost ascending
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (-c.quality, c.cost.llm_token_cost + c.cost.wall_time_sec),
        )

        pareto: List[SearchResult] = []
        min_cost = float("inf")

        for candidate in sorted_candidates:
            total_cost = candidate.cost.llm_token_cost + candidate.cost.wall_time_sec
            # This candidate is on Pareto front if no previous candidate
            # has both higher quality AND lower cost
            if total_cost < min_cost:
                pareto.append(candidate)
                min_cost = total_cost

        return pareto

    def _find_best_by_quality(self, candidates: List[SearchResult]) -> Optional[SearchResult]:
        """Find candidate with highest quality."""
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.quality)

    def _find_best_by_cost(self, candidates: List[SearchResult]) -> Optional[SearchResult]:
        """Find candidate with lowest cost."""
        if not candidates:
            return None
        return min(candidates, key=lambda c: c.cost.llm_token_cost + c.cost.wall_time_sec)

    def _find_best_balanced(
        self,
        candidates: List[SearchResult],
        quality_weight: float = 0.5,
    ) -> Optional[SearchResult]:
        """
        Find best balanced candidate using normalized scores.

        Score = quality_weight * normalized_quality +
                (1 - quality_weight) * normalized_cost_inverse
        """
        if not candidates:
            return None

        # Normalize quality (0-1)
        qualities = [c.quality for c in candidates]
        q_min, q_max = min(qualities), max(qualities)
        q_range = q_max - q_min if q_max > q_min else 1.0

        # Normalize cost (inverse, so lower cost = higher score)
        costs = [c.cost.llm_token_cost + c.cost.wall_time_sec for c in candidates]
        c_min, c_max = min(costs), max(costs)
        c_range = c_max - c_min if c_max > c_min else 1.0

        def score(c: SearchResult) -> float:
            norm_q = (c.quality - q_min) / q_range
            total_cost = c.cost.llm_token_cost + c.cost.wall_time_sec
            norm_c = 1.0 - (total_cost - c_min) / c_range
            return quality_weight * norm_q + (1 - quality_weight) * norm_c

        return max(candidates, key=score)


__all__ = [
    "SearchStrategyType",
    "OptimizationObjective",
    "SearchResult",
    "SearchReport",
    "SearchConfig",
    "SearchStrategy",
    "BaseSearchStrategy",
]