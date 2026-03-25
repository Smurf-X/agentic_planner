# -*- coding: utf-8 -*-
"""
Enhanced Beam search strategy with action-based optimization.

Beam search maintains top-k candidates at each iteration and explores
their neighbors. This enhanced version supports:
- Action-based optimization (operator, directive) pairs
- LLM-guided action selection with rich context
- Multi-objective optimization (quality vs cost)
- Pareto frontier tracking
- Adaptive beam width
- Action statistics tracking across search
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from agentic_planner.contracts.cost import CostBreakdown
from agentic_planner.contracts.recipe import DJExecutableConfig, validate_executable_config
from agentic_planner.optimizer.action import Action, ActionSpace, ActionSpaceBuilder
from agentic_planner.optimizer.action_context import (
    ActionSelectionContext,
    ActionStatsTracker,
    TriedAction,
)
from agentic_planner.optimizer.directives.base import DirectiveResult
from agentic_planner.optimizer.directives.registry import DIRECTIVE_REGISTRY
from agentic_planner.optimizer.search.base import (
    BaseSearchStrategy,
    SearchConfig,
    SearchReport,
    SearchResult,
    SearchStrategyType,
)

if TYPE_CHECKING:
    from agentic_planner.optimizer.llm_action_selector import LLMActionSelector


class BeamSearchConfig(BaseModel):
    """Configuration for beam search."""

    beam_width: int = Field(default=4, ge=1, le=64)
    max_iterations: int = Field(default=3, ge=1, le=50)

    expansion_directives: List[str] = Field(
        default_factory=list,
        description="Directive names to include in action space. Empty = use all registered.",
    )

    use_llm_selection: bool = Field(
        default=True,
        description="Use LLM to select promising actions.",
    )

    llm_selection_top_k: int = Field(
        default=10,
        description="Number of actions for LLM to select.",
    )

    optimize_goal: str = Field(
        default="quality",
        description="Optimization goal: 'quality', 'cost', or 'balanced'.",
    )

    track_pareto: bool = Field(default=True, description="Track Pareto frontier during search.")
    cost_weight: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Weight for cost in ranking (0=quality only)."
    )
    adaptive_beam: bool = Field(
        default=False,
        description="Adaptively adjust beam width based on improvement.",
    )
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
    used_actions: Set[Tuple[int, str]] = field(default_factory=set)
    tried_actions: List[TriedAction] = field(default_factory=list)
    execution_sample: List[Dict[str, Any]] = field(default_factory=list)

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
        import math

        quality_score = self.quality
        total_cost = self.cost.llm_token_cost + self.cost.wall_time_sec
        cost_score = 1.0 / (1.0 + math.log1p(total_cost))
        return (1 - cost_weight) * quality_score + cost_weight * cost_score

    def get_total_cost(self) -> float:
        """Get total cost as a single number."""
        return self.cost.llm_token_cost + self.cost.wall_time_sec


class BeamSearchStrategy(BaseSearchStrategy):
    """
    Enhanced beam search with action-based optimization.

    Features:
    - Action-based: (operator, directive) pairs
    - LLM-guided action selection with rich context
    - Maintains top-k candidates at each iteration
    - Tracks Pareto frontier across all iterations
    - Supports cost-weighted ranking
    - Optional adaptive beam width
    - Action statistics tracking for LLM guidance
    """

    def __init__(
        self,
        config: BeamSearchConfig,
        evaluator: Optional[Any] = None,
        llm_selector: Optional[LLMActionSelector] = None,
        action_builder: Optional[ActionSpaceBuilder] = None,
    ) -> None:
        super().__init__(
            SearchConfig(strategy=SearchStrategyType.BEAM),
            evaluator,
        )
        self._beam_config = config
        self._llm_selector = llm_selector
        self._rng = random.Random(config.seed)
        self._seen_hashes: Set[str] = set()
        self._action_stats = ActionStatsTracker()
        self._data_sample: List[Dict[str, Any]] = []

        if action_builder:
            self._action_builder = action_builder
        else:
            directives = None
            if config.expansion_directives:
                directives = [
                    DIRECTIVE_REGISTRY[name]
                    for name in config.expansion_directives
                    if name in DIRECTIVE_REGISTRY
                ]
            self._action_builder = ActionSpaceBuilder(directives=directives)

    def set_llm_selector(self, selector: LLMActionSelector) -> None:
        """Set the LLM action selector."""
        self._llm_selector = selector

    def set_data_sample(self, sample: List[Dict[str, Any]]) -> None:
        """Set data sample for LLM context."""
        self._data_sample = sample

    def search(self, root: DJExecutableConfig) -> SearchReport:
        """Execute beam search."""
        errors = validate_executable_config(root)
        if errors:
            return SearchReport(
                ok=False,
                candidates=[],
                errors=["Invalid root config: " + "; ".join(errors)],
            )

        if self._evaluator is not None and hasattr(self._evaluator, "prepare_fixed_samples"):
            dataset_path = root.get("dataset_path")
            self._evaluator.prepare_fixed_samples(dataset_path)

        all_candidates: List[SearchResult] = []
        pareto_candidates: List[SearchResult] = []

        root_result = self._evaluate_full(root)
        root_outputs = (
            root_result.outputs
            if hasattr(root_result, "outputs") and isinstance(root_result.outputs, list)
            else []
        )
        root_beam = BeamCandidate(
            config=deepcopy(root),
            cost=root_result.cost,
            quality=root_result.quality,
            origin="root",
            trace=[],
            used_actions=set(),
            execution_sample=root_outputs[:5],
        )
        self._seen_hashes.add(root_beam.config_hash)

        beams: List[BeamCandidate] = [root_beam]
        all_candidates.append(self._beam_to_result(root_beam, 0))

        current_beam_width = self._beam_config.beam_width
        last_best_score = root_beam.score(self._beam_config.cost_weight)

        base_action_space = self._action_builder.build(root)

        for iteration in range(self._beam_config.max_iterations):
            self._iteration_count = iteration + 1
            next_beams: List[BeamCandidate] = []

            for beam in beams:
                available_actions = self._get_available_actions(beam, base_action_space)

                if not available_actions:
                    continue

                actions_to_try = self._select_actions(
                    available_actions,
                    beam,
                    iteration,
                    all_candidates,
                )

                for action in actions_to_try:
                    if self._evaluated_count >= self._config.max_evaluations:
                        break

                    step = action.apply(beam.config)

                    if not step.ok or step.config_after is None:
                        continue
                    if not step.applied:
                        continue

                    val_errors = validate_executable_config(step.config_after)
                    if val_errors:
                        continue

                    child_config = step.config_after
                    child_hash = BeamCandidate._hash_config(child_config)

                    if self._beam_config.deduplicate and child_hash in self._seen_hashes:
                        continue
                    self._seen_hashes.add(child_hash)

                    child_result = self._evaluate_full(child_config)

                    self._action_stats.record(
                        action=action,
                        before_quality=beam.quality,
                        before_cost=beam.get_total_cost(),
                        after_quality=child_result.quality,
                        after_cost=child_result.cost.llm_token_cost
                        + child_result.cost.wall_time_sec,
                    )

                    tried = TriedAction(
                        action=action,
                        result_quality=child_result.quality,
                        result_cost=child_result.cost.llm_token_cost
                        + child_result.cost.wall_time_sec,
                        quality_change=child_result.quality - beam.quality,
                        cost_change=(
                            child_result.cost.llm_token_cost + child_result.cost.wall_time_sec
                        )
                        - beam.get_total_cost(),
                        applied=True,
                    )

                    new_used = beam.used_actions.copy()
                    new_used.add((action.operator_index, action.directive_name))

                    child_outputs = (
                        child_result.outputs
                        if hasattr(child_result, "outputs")
                        and isinstance(child_result.outputs, list)
                        else []
                    )
                    child = BeamCandidate(
                        config=child_config,
                        cost=child_result.cost,
                        quality=child_result.quality,
                        origin=f"{beam.origin}+{action.directive_name}[{action.operator_name}]",
                        trace=beam.trace + [step],
                        used_actions=new_used,
                        tried_actions=beam.tried_actions + [tried],
                        execution_sample=child_outputs[:5],
                    )
                    next_beams.append(child)
                    all_candidates.append(self._beam_to_result(child, iteration + 1))

                if self._evaluated_count >= self._config.max_evaluations:
                    break

            if not next_beams:
                break

            next_beams.sort(key=lambda b: b.score(self._beam_config.cost_weight), reverse=True)

            if self._beam_config.adaptive_beam and next_beams:
                best_score = next_beams[0].score(self._beam_config.cost_weight)
                improvement = best_score - last_best_score
                if improvement > 0.01:
                    current_beam_width = min(
                        current_beam_width + 2, self._beam_config.max_beam_width
                    )
                elif improvement < 0.001:
                    current_beam_width = max(
                        current_beam_width - 1, self._beam_config.min_beam_width
                    )
                last_best_score = best_score

            beams = next_beams[:current_beam_width]

            if self._beam_config.track_pareto:
                pareto_candidates = self._compute_pareto_front(all_candidates)

        pareto = (
            self._compute_pareto_front(all_candidates)
            if not self._beam_config.track_pareto
            else pareto_candidates
        )

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
                "action_space_size": len(base_action_space),
            },
        )

    def _get_available_actions(
        self,
        beam: BeamCandidate,
        base_space: ActionSpace,
    ) -> List[Action]:
        """Get available actions for a beam, excluding already-used ones."""
        return [
            a for a in base_space if (a.operator_index, a.directive_name) not in beam.used_actions
        ]

    def _determine_optimize_goal(
        self, beam: BeamCandidate, all_candidates: List[SearchResult]
    ) -> str:
        """
        Determine optimization goal based on current state.

        Similar to docetl: if quality is above median, optimize for cost;
        otherwise optimize for quality.
        """
        if not all_candidates:
            return "quality"

        qualities = [c.quality for c in all_candidates if c.quality > 0]
        if not qualities:
            return "quality"

        qualities.sort()
        median_quality = qualities[len(qualities) // 2]

        if beam.quality > median_quality:
            return "cost"
        return "quality"

    def _select_actions(
        self,
        available_actions: List[Action],
        beam: BeamCandidate,
        iteration: int,
        all_candidates: Optional[List[SearchResult]] = None,
    ) -> List[Action]:
        """Select actions to try for a beam."""
        if self._beam_config.use_llm_selection and self._llm_selector:
            optimize_goal = self._determine_optimize_goal(beam, all_candidates or [])

            context = ActionSelectionContext(
                config=beam.config,
                quality=beam.quality,
                cost=beam.get_total_cost(),
                available_actions=available_actions,
                execution_sample=beam.execution_sample,
                data_sample=self._data_sample,
                action_stats=self._action_stats,
                tried_actions=beam.tried_actions,
                optimize_goal=optimize_goal,
                iteration=iteration,
                max_iterations=self._beam_config.max_iterations,
            )

            from agentic_planner.optimizer.action import ActionSpace

            temp_space = ActionSpace(
                actions=available_actions,
                config_hash="",
                operator_count=0,
            )

            result = self._llm_selector.select_with_context(
                temp_space,
                context,
                top_k=self._beam_config.llm_selection_top_k,
            )
            return result.selected_actions

        return available_actions[: self._beam_config.llm_selection_top_k]

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
        return [
            type(
                "CandidateRecord",
                (),
                {
                    "config": c.config,
                    "cost": c.cost,
                    "quality": c.quality,
                    "origin": c.origin,
                    "trace": c.trace,
                },
            )()
            for c in report.candidates
        ]


__all__ = [
    "BeamSearchConfig",
    "BeamSearchStrategy",
    "BeamSearchOptimizer",
    "BeamCandidate",
]
