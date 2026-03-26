# -*- coding: utf-8 -*-
"""MOAR search skeleton for staged migration baseline."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from pydantic import BaseModel, Field

from agentic_planner.contracts.recipe import DJExecutableConfig
from agentic_planner.optimizer.search.base import (
    BaseSearchStrategy,
    SearchConfig,
    SearchReport,
    SearchResult,
    SearchStrategyType,
)


class MOARSearchConfig(BaseModel):
    """Configuration for MOAR search."""

    strategy: SearchStrategyType = Field(
        default=SearchStrategyType.MCTS,
        description="MOAR strategy marker for this staged migration.",
    )
    max_iterations: int = Field(default=3, ge=1, le=200)
    max_evaluations: int = Field(default=100, ge=1, le=10000)
    exploration_weight: float = Field(default=1.4, ge=0.0)
    seed: int = Field(default=42)

    model_config = {"extra": "allow"}


class MOARSearchStrategy(BaseSearchStrategy):
    """Minimal MOAR strategy placeholder used by the optimization runner."""

    def __init__(
        self,
        config: MOARSearchConfig,
        evaluator: Optional[Any] = None,
    ) -> None:
        super().__init__(
            SearchConfig(
                strategy=config.strategy,
                max_iterations=config.max_iterations,
                max_evaluations=config.max_evaluations,
                seed=config.seed,
            ),
            evaluator,
        )
        self._moar_config = config

    def search(self, root: DJExecutableConfig) -> SearchReport:
        """Execute minimal MOAR search by evaluating only the root."""
        cost, quality = self._evaluate(root)
        if self._evaluator is None:
            self._evaluated_count = 1

        self._iteration_count = 1
        root_result = SearchResult(
            config=deepcopy(root),
            cost=cost,
            quality=quality,
            origin="root",
            generation=0,
        )

        return SearchReport(
            ok=True,
            candidates=[root_result],
            pareto_front=[root_result],
            total_iterations=self._iteration_count,
            total_evaluations=self._evaluated_count,
            best_by_quality=root_result,
            best_by_cost=root_result,
            best_balanced=root_result,
            metrics={"strategy": "moar_root_only"},
        )


__all__ = ["MOARSearchConfig", "MOARSearchStrategy"]
