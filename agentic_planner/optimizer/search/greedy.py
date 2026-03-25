# -*- coding: utf-8 -*-
"""
Greedy search strategy for pipeline optimization.

Greedy search always picks the best immediate improvement at each step.
Simple but can get stuck in local optima.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agentic_planner.contracts.recipe import DJExecutableConfig, validate_executable_config
from agentic_planner.optimizer.directives.base import DirectiveResult
from agentic_planner.optimizer.directives.registry import DIRECTIVE_REGISTRY
from agentic_planner.optimizer.search.base import (
    BaseSearchStrategy,
    SearchConfig,
    SearchReport,
    SearchResult,
    SearchStrategyType,
)


class GreedySearchConfig(BaseModel):
    """Configuration for greedy search."""

    expansion_directives: List[str] = Field(
        default_factory=list,
        description="Directive keys to try as expansion moves.",
    )
    max_iterations: int = Field(default=10, ge=1, le=100)
    max_no_improve: int = Field(default=3, description="Stop after N iterations without improvement.")
    minimize_cost: bool = Field(default=False, description="Minimize cost instead of maximizing quality.")
    randomize_order: bool = Field(default=False, description="Randomize directive evaluation order.")

    model_config = {"extra": "allow"}


class GreedySearchStrategy(BaseSearchStrategy):
    """
    Greedy hill-climbing search.

    At each step, try all expansion directives and pick the one with
    the best improvement. Continue until no improvement or max iterations.
    """

    def __init__(
        self,
        config: GreedySearchConfig,
        evaluator: Optional[Any] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            SearchConfig(strategy=SearchStrategyType.GREEDY),
            evaluator,
        )
        self._greedy_config = config
        self._rng = random.Random(seed)

    def search(self, root: DJExecutableConfig) -> SearchReport:
        """Execute greedy hill-climbing search."""
        errors = validate_executable_config(root)
        if errors:
            return SearchReport(
                ok=False,
                candidates=[],
                errors=["Invalid root config: " + "; ".join(errors)],
            )

        candidates: List[SearchResult] = []
        all_evaluated: List[SearchResult] = []

        # Evaluate root
        root_cost, root_quality = self._evaluate(root)
        current = SearchResult(
            config=deepcopy(root),
            cost=root_cost,
            quality=root_quality,
            origin="root",
            generation=0,
        )
        candidates.append(current)
        all_evaluated.append(current)

        best_quality = root_quality
        best_cost = root_cost.llm_token_cost + root_cost.wall_time_sec
        no_improve_count = 0
        trace: List[DirectiveResult] = []

        for iteration in range(self._greedy_config.max_iterations):
            self._iteration_count = iteration + 1

            # Try all directives
            directive_names = list(self._greedy_config.expansion_directives)
            if self._greedy_config.randomize_order:
                self._rng.shuffle(directive_names)

            best_neighbor: Optional[SearchResult] = None
            best_directive: Optional[str] = None
            best_step: Optional[DirectiveResult] = None

            for dname in directive_names:
                directive = DIRECTIVE_REGISTRY.get(dname)
                if directive is None:
                    continue

                step = directive.apply(current.config)
                if not step.ok or step.config_after is None:
                    continue
                if not step.applied:
                    continue
                if validate_executable_config(step.config_after):
                    continue

                neighbor_cost, neighbor_quality = self._evaluate(step.config_after)
                neighbor = SearchResult(
                    config=step.config_after,
                    cost=neighbor_cost,
                    quality=neighbor_quality,
                    origin=f"{current.origin}+{dname}",
                    generation=iteration + 1,
                    parent_id=str(id(current)),
                    trace=[step],
                )
                all_evaluated.append(neighbor)

                # Check if this is better
                is_better = False
                if self._greedy_config.minimize_cost:
                    neighbor_total = neighbor_cost.llm_token_cost + neighbor_cost.wall_time_sec
                    is_better = neighbor_quality >= current.quality and neighbor_total < best_cost
                else:
                    is_better = neighbor_quality > current.quality

                if is_better:
                    if best_neighbor is None or (
                        self._greedy_config.minimize_cost
                        and neighbor_quality >= best_neighbor.quality
                        or neighbor_quality > best_neighbor.quality
                    ):
                        best_neighbor = neighbor
                        best_directive = dname
                        best_step = step

            if best_neighbor is None:
                no_improve_count += 1
                if no_improve_count >= self._greedy_config.max_no_improve:
                    break
            else:
                current = best_neighbor
                candidates.append(best_neighbor)
                if best_step:
                    trace.append(best_step)

                if self._greedy_config.minimize_cost:
                    best_cost = best_neighbor.cost.llm_token_cost + best_neighbor.cost.wall_time_sec
                else:
                    best_quality = best_neighbor.quality
                no_improve_count = 0

            if self._evaluated_count >= self._config.max_evaluations:
                break

        # Compute Pareto front and best candidates
        pareto = self._compute_pareto_front(all_evaluated)

        return SearchReport(
            ok=True,
            candidates=all_evaluated,
            pareto_front=pareto,
            total_iterations=self._iteration_count,
            total_evaluations=self._evaluated_count,
            best_by_quality=self._find_best_by_quality(all_evaluated),
            best_by_cost=self._find_best_by_cost(all_evaluated),
            best_balanced=self._find_best_balanced(all_evaluated),
            metrics={
                "improvements": len(candidates) - 1,
                "final_quality": current.quality,
                "final_cost": current.cost.llm_token_cost,
            },
        )


__all__ = ["GreedySearchConfig", "GreedySearchStrategy"]