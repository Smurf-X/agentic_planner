# -*- coding: utf-8 -*-
"""
Random search strategy for pipeline optimization.

Random search samples configurations uniformly at random.
Good baseline and can sometimes find good solutions surprisingly effectively.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agentic_planner.contracts.recipe import DJExecutableConfig, validate_executable_config
from agentic_planner.optimizer.directives import DIRECTIVE_REGISTRY
from agentic_planner.optimizer.search.base import (
    BaseSearchStrategy,
    SearchConfig,
    SearchReport,
    SearchResult,
    SearchStrategyType,
)


class RandomSearchConfig(BaseModel):
    """Configuration for random search."""

    expansion_directives: List[str] = Field(
        default_factory=list,
        description="Directive keys to sample from.",
    )
    max_samples: int = Field(default=50, ge=1, le=500, description="Maximum samples to evaluate.")
    max_depth: int = Field(default=5, ge=1, le=20, description="Maximum chain depth per sample.")
    keep_all: bool = Field(
        default=True, description="Keep all evaluated samples, not just improvements."
    )

    model_config = {"extra": "allow"}


class RandomSearchStrategy(BaseSearchStrategy):
    """
    Random sampling search.

    Generate random configurations by applying random directive sequences.
    Evaluate each and keep track of the best found.
    """

    def __init__(
        self,
        config: RandomSearchConfig,
        evaluator: Optional[Any] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            SearchConfig(strategy=SearchStrategyType.RANDOM),
            evaluator,
        )
        self._random_config = config
        self._rng = random.Random(seed)

    def search(self, root: DJExecutableConfig) -> SearchReport:
        """Execute random search."""
        errors = validate_executable_config(root)
        if errors:
            return SearchReport(
                ok=False,
                candidates=[],
                errors=["Invalid root config: " + "; ".join(errors)],
            )

        all_evaluated: List[SearchResult] = []

        # Evaluate root
        root_cost, root_quality = self._evaluate(root)
        root_result = SearchResult(
            config=deepcopy(root),
            cost=root_cost,
            quality=root_quality,
            origin="root",
            generation=0,
        )
        all_evaluated.append(root_result)

        # Get valid directives
        valid_directives = [
            d for d in self._random_config.expansion_directives if d in DIRECTIVE_REGISTRY
        ]
        if not valid_directives:
            return SearchReport(
                ok=True,
                candidates=all_evaluated,
                pareto_front=[root_result],
                total_iterations=0,
                total_evaluations=1,
                best_by_quality=root_result,
                best_by_cost=root_result,
                best_balanced=root_result,
            )

        # Sample configurations
        for sample_idx in range(self._random_config.max_samples):
            if self._evaluated_count >= self._config.max_evaluations:
                break

            # Generate a random configuration by applying random directives
            current_config = deepcopy(root)
            applied_trace = []
            depth = self._rng.randint(1, self._random_config.max_depth)

            for step_idx in range(depth):
                dname = self._rng.choice(valid_directives)
                directive = DIRECTIVE_REGISTRY[dname]

                step = directive.apply(current_config)
                if step.ok and step.applied and step.config_after:
                    if not validate_executable_config(step.config_after):
                        current_config = step.config_after
                        applied_trace.append(step)

            # Evaluate the generated configuration
            config_cost, config_quality = self._evaluate(current_config)
            result = SearchResult(
                config=current_config,
                cost=config_cost,
                quality=config_quality,
                origin=f"random_{sample_idx}_d{len(applied_trace)}",
                generation=len(applied_trace),
                trace=applied_trace,
            )
            all_evaluated.append(result)

        # Compute Pareto front and best candidates
        pareto = self._compute_pareto_front(all_evaluated)

        return SearchReport(
            ok=True,
            candidates=all_evaluated,
            pareto_front=pareto,
            total_iterations=self._random_config.max_samples,
            total_evaluations=self._evaluated_count,
            best_by_quality=self._find_best_by_quality(all_evaluated),
            best_by_cost=self._find_best_by_cost(all_evaluated),
            best_balanced=self._find_best_balanced(all_evaluated),
            metrics={
                "samples_generated": len(all_evaluated) - 1,
                "pareto_size": len(pareto),
            },
        )


__all__ = ["RandomSearchConfig", "RandomSearchStrategy"]
