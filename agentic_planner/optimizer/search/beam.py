# -*- coding: utf-8 -*-
"""
Enhanced Beam search strategy with multi-objective support.

Beam search maintains top-k candidates at each iteration and explores
their neighbors. This enhanced version supports:
- Multi-objective optimization (quality vs cost)
- Pareto frontier tracking
- Adaptive beam width
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from agentic_planner.contracts.cost import CostBreakdown
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


class BeamSearchConfig(BaseModel):
    """Configuration for beam search."""

    beam_width: int = Field(default=4, ge=1, le=64)
    max_iterations: int = Field(default=3, ge=1, le=50)
    expansion_directives: List[str] = Field(
        default_factory=list,
        description="Directive keys tried as one-step neighbors from each beam.",
    )
    # Multi-objective settings
    track_pareto: bool = Field(default=True, description="Track Pareto frontier during search.")
    cost_weight: float = Field(default=0.0, ge=0.0, le=1.0, description="Weight for cost in ranking (0=quality only).")
    adaptive_beam: bool = Field(default=False, description="Adaptively adjust beam width based on improvement.")
    min_beam_width: int = Field(default=2, ge=1, description="Minimum beam width when adaptive.")
    max_beam_width: int = Field(default=16, ge=1, description="Maximum beam width when adaptive.")
    deduplicate: bool = Field(default=True, description="Remove duplicate configurations.")
    seed: int = Field(default=42)

    model_config = {"extra": "allow"}


@dataclass
class BeamCandidate:
    """Internal representation of a beam candidate."""

    config: DJExecutableConfig
    cost: CostBreakdown
    quality: float
    origin: str
    trace: List[DirectiveResult]
    config_hash: str = ""

    def __post_init__(self):
        if not self.config_hash:
            self.config_hash = self._hash_config(self.config)

    @staticmethod
    def _hash_config(cfg: DJExecutableConfig) -> str:
        """Generate a hash for configuration deduplication."""
        import hashlib
        import json
        content = json.dumps(cfg.get("process", []), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def score(self, cost_weight: float = 0.0) -> float:
        """Compute a combined score for ranking."""
        # Normalize quality to 0-1, cost inverse to 0-1
        # Higher score is better
        quality_score = self.quality  # Assume already 0-1
        total_cost = self.cost.llm_token_cost + self.cost.wall_time_sec
        # Use log scale for cost to handle large differences
        import math
        cost_score = 1.0 / (1.0 + math.log1p(total_cost))
        return (1 - cost_weight) * quality_score + cost_weight * cost_score


class BeamSearchStrategy(BaseSearchStrategy):
    """
    Enhanced beam search with multi-objective optimization.

    Features:
    - Maintains top-k candidates at each iteration
    - Tracks Pareto frontier across all iterations
    - Supports cost-weighted ranking
    - Optional adaptive beam width
    """

    def __init__(
        self,
        config: BeamSearchConfig,
        evaluator: Optional[Any] = None,
    ) -> None:
        super().__init__(
            SearchConfig(strategy=SearchStrategyType.BEAM),
            evaluator,
        )
        self._beam_config = config
        self._rng = random.Random(config.seed)
        self._seen_hashes: Set[str] = set()

    def search(self, root: DJExecutableConfig) -> SearchReport:
        """Execute beam search."""
        errors = validate_executable_config(root)
        if errors:
            return SearchReport(
                ok=False,
                candidates=[],
                errors=["Invalid root config: " + "; ".join(errors)],
            )

        all_candidates: List[SearchResult] = []
        pareto_candidates: List[SearchResult] = []

        # Evaluate root
        root_cost, root_quality = self._evaluate(root)
        root_beam = BeamCandidate(
            config=deepcopy(root),
            cost=root_cost,
            quality=root_quality,
            origin="root",
            trace=[],
        )
        self._seen_hashes.add(root_beam.config_hash)

        beams: List[BeamCandidate] = [root_beam]
        all_candidates.append(self._beam_to_result(root_beam, 0))

        current_beam_width = self._beam_config.beam_width
        last_best_score = root_beam.score(self._beam_config.cost_weight)

        for iteration in range(self._beam_config.max_iterations):
            self._iteration_count = iteration + 1
            next_beams: List[BeamCandidate] = []

            for beam in beams:
                for dname in self._beam_config.expansion_directives:
                    if self._evaluated_count >= self._config.max_evaluations:
                        break

                    directive = DIRECTIVE_REGISTRY.get(dname)
                    if directive is None:
                        continue

                    step = directive.apply(beam.config)
                    if not step.ok or step.config_after is None:
                        continue
                    if not step.applied:
                        continue
                    if validate_executable_config(step.config_after):
                        continue

                    child_config = step.config_after
                    child_hash = BeamCandidate._hash_config(child_config)

                    # Deduplication
                    if self._beam_config.deduplicate and child_hash in self._seen_hashes:
                        continue
                    self._seen_hashes.add(child_hash)

                    # Evaluate
                    child_cost, child_quality = self._evaluate(child_config)
                    child = BeamCandidate(
                        config=child_config,
                        cost=child_cost,
                        quality=child_quality,
                        origin=f"{beam.origin}+{dname}",
                        trace=beam.trace + [step],
                    )
                    next_beams.append(child)
                    all_candidates.append(self._beam_to_result(child, iteration + 1))

                if self._evaluated_count >= self._config.max_evaluations:
                    break

            if not next_beams:
                break

            # Sort by score and select top beams
            next_beams.sort(key=lambda b: b.score(self._beam_config.cost_weight), reverse=True)

            # Adaptive beam width
            if self._beam_config.adaptive_beam and next_beams:
                best_score = next_beams[0].score(self._beam_config.cost_weight)
                improvement = best_score - last_best_score
                if improvement > 0.01:  # Significant improvement
                    current_beam_width = min(
                        current_beam_width + 2,
                        self._beam_config.max_beam_width
                    )
                elif improvement < 0.001:  # Little improvement
                    current_beam_width = max(
                        current_beam_width - 1,
                        self._beam_config.min_beam_width
                    )
                last_best_score = best_score

            beams = next_beams[:current_beam_width]

            # Update Pareto frontier
            if self._beam_config.track_pareto:
                pareto_candidates = self._compute_pareto_front(all_candidates)

        # Final Pareto computation
        pareto = self._compute_pareto_front(all_candidates) if not self._beam_config.track_pareto else pareto_candidates

        return SearchReport(
            ok=True,
            candidates=all_candidates,
            pareto_front=pareto,
            total_iterations=self._iteration_count,
            total_evaluations=self._evaluated_count,
            best_by_quality=self._find_best_by_quality(all_candidates),
            best_by_cost=self._find_best_by_cost(all_candidates),
            best_balanced=self._find_best_balanced(all_candidates),
            metrics={
                "beam_width": current_beam_width,
                "unique_configs": len(self._seen_hashes),
                "pareto_size": len(pareto),
            },
        )

    def _beam_to_result(self, beam: BeamCandidate, generation: int) -> SearchResult:
        """Convert BeamCandidate to SearchResult."""
        return SearchResult(
            config=beam.config,
            cost=beam.cost,
            quality=beam.quality,
            origin=beam.origin,
            trace=beam.trace,
            generation=generation,
        )


# Legacy compatibility: keep the old BeamSearchOptimizer interface
class BeamSearchOptimizer:
    """
    Legacy interface for beam search.

    Maintains backward compatibility with existing code.
    """

    def __init__(
        self,
        beam_config: BeamSearchConfig,
        evaluator: Optional[Any] = None,
        eval_config: Optional[Any] = None,
    ) -> None:
        self._strategy = BeamSearchStrategy(beam_config, evaluator)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeamSearchOptimizer":
        return cls(BeamSearchConfig.model_validate(data))

    def search(self, root: DJExecutableConfig) -> List[Any]:
        """
        Legacy search interface.

        Returns list of CandidateRecord for backward compatibility.
        """
        report = self._strategy.search(root)
        # Convert SearchResult to CandidateRecord-like objects
        return [
            type("CandidateRecord", (), {
                "config": c.config,
                "cost": c.cost,
                "quality": c.quality,
                "origin": c.origin,
                "trace": c.trace,
            })()
            for c in report.candidates
        ]


__all__ = [
    "BeamSearchConfig",
    "BeamSearchStrategy",
    "BeamSearchOptimizer",
    "BeamCandidate",
]