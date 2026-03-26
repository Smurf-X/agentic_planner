# -*- coding: utf-8 -*-
"""MOAR search with a minimal but real tree-search loop."""

from __future__ import annotations

from copy import deepcopy
import math
import random
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from agentic_planner.contracts.recipe import DJExecutableConfig
from agentic_planner.optimizer.action import ActionSpaceBuilder
from agentic_planner.optimizer.directives.base import DirectiveResult
from agentic_planner.optimizer.directives.registry import list_directive_spec_names
from agentic_planner.optimizer.model_registry import ModelRegistry
from agentic_planner.optimizer.op_locator import ProcessIndex
from agentic_planner.optimizer.search.base import (
    BaseSearchStrategy,
    SearchConfig,
    SearchReport,
    SearchResult,
    SearchStrategyType,
)
from agentic_planner.optimizer.search.pareto import ParetoFrontier
from agentic_planner.optimizer.search.tree_node import SearchTreeNode


class MOARSearchConfig(BaseModel):
    """Configuration for MOAR search."""

    strategy: Literal[SearchStrategyType.MCTS] = Field(
        default=SearchStrategyType.MCTS,
        description="MOAR strategy marker for this staged migration.",
    )
    max_iterations: int = Field(default=3, ge=1, le=200)
    max_evaluations: int = Field(default=100, ge=1, le=10000)
    max_tree_depth: int = Field(default=4, ge=1, le=100)
    exploration_weight: float = Field(default=1.4, ge=0.0)
    early_stop_patience: int = Field(default=3, ge=1, le=200)
    frontier_improvement_threshold: float = Field(default=1.0, ge=0.0)
    directive_names: List[str] = Field(default_factory=list)
    allowed_safety_levels: List[str] = Field(default_factory=lambda: ["safe"])
    seed: int = Field(default=42)

    model_config = {"extra": "allow"}


class MOARSearchStrategy(BaseSearchStrategy):
    """MOAR strategy with select/expand/simulate/backpropagate tree search."""

    def __init__(
        self,
        config: MOARSearchConfig,
        evaluator: Optional[Any] = None,
        model_registry: Optional[ModelRegistry] = None,
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
        self._model_registry = model_registry or ModelRegistry.default()
        directive_names = config.directive_names or list_directive_spec_names()
        self._action_builder = ActionSpaceBuilder(
            directive_names=directive_names,
            model_registry=self._model_registry,
            allowed_safety_levels=config.allowed_safety_levels,
        )
        self._rng = random.Random(config.seed)
        self._node_counter = 0

    def search(self, root: DJExecutableConfig) -> SearchReport:
        """Execute MOAR search over dynamic per-node actions."""
        self._iteration_count = 0
        self._evaluated_count = 0
        self._node_counter = 0

        root_node = SearchTreeNode.from_config(
            root,
            parent_id=None,
            node_id=self._next_node_id(),
            depth=0,
        )

        frontier = ParetoFrontier()
        candidates: List[SearchResult] = []

        root_result = self._evaluate_node(root_node, origin="root", generation=0)
        root_frontier = frontier.add_with_details(
            quality=root_result.quality,
            total_cost=self._total_cost(root_result),
            candidate_id=root_node.node_id,
            payload=root_result,
        )
        candidates.append(root_result)
        self._backpropagate([root_node], root_frontier.reward)
        self._seed_root_model_baselines(root_node, frontier, candidates)

        no_improvement_streak = 0
        early_stopped = False
        stop_reason = "max_iterations"

        for iteration in range(self._moar_config.max_iterations):
            if self._evaluated_count >= self._moar_config.max_evaluations:
                stop_reason = "max_evaluations"
                break

            selection_path = self._select_path(root_node)
            leaf = selection_path[-1]
            expanded_node = self._expand(leaf)
            if expanded_node is None:
                no_improvement_streak += 1
                self._iteration_count = iteration + 1
                if no_improvement_streak >= self._moar_config.early_stop_patience:
                    early_stopped = True
                    stop_reason = "no_expandable_actions"
                    break
                continue

            rollout_path = list(selection_path)
            rollout_path.append(expanded_node)
            result = self._simulate(expanded_node)
            candidates.append(result)

            frontier_update = frontier.add_with_details(
                quality=result.quality,
                total_cost=self._total_cost(result),
                candidate_id=expanded_node.node_id,
                payload=result,
            )
            self._backpropagate(rollout_path, frontier_update.reward)

            if frontier_update.reward >= self._moar_config.frontier_improvement_threshold:
                no_improvement_streak = 0
            else:
                no_improvement_streak += 1

            self._iteration_count = iteration + 1
            if no_improvement_streak >= self._moar_config.early_stop_patience:
                early_stopped = True
                stop_reason = "frontier_improvement_stalled"
                break

        if self._iteration_count == 0:
            self._iteration_count = 1

        pareto_front: List[SearchResult] = [
            member.payload for member in frontier.members if member.payload is not None
        ]

        best_by_quality = self._find_best_by_quality(candidates)
        best_by_cost = self._find_best_by_cost(candidates)
        best_balanced = self._find_best_balanced(candidates)

        return SearchReport(
            ok=True,
            candidates=candidates,
            pareto_front=pareto_front,
            total_iterations=self._iteration_count,
            total_evaluations=self._evaluated_count,
            best_by_quality=best_by_quality,
            best_by_cost=best_by_cost,
            best_balanced=best_balanced,
            metrics={
                "strategy": "moar_tree_search",
                "frontier_size": len(pareto_front),
                "early_stopped": early_stopped,
                "stop_reason": stop_reason,
                "no_improvement_streak": no_improvement_streak,
                "root_frontier_reward": root_frontier.reward,
            },
        )

    def _next_node_id(self) -> str:
        """Allocate stable node ids for deterministic tracing."""
        node_id = f"node-{self._node_counter}"
        self._node_counter += 1
        return node_id

    def _evaluate_config(self, config: DJExecutableConfig) -> Tuple[Any, float]:
        """Evaluate a config while tracking fallback-evaluator counts."""
        cost, quality = self._evaluate(config)
        if self._evaluator is None:
            self._evaluated_count += 1
        return cost, quality

    def _evaluate_node(self, node: SearchTreeNode, origin: str, generation: int) -> SearchResult:
        """Evaluate a node and package a SearchResult."""
        cost, quality = self._evaluate_config(node.config)
        incoming_step = getattr(node, "incoming_directive_result", None)
        trace: List[Any] = []
        if incoming_step is not None:
            trace.append(incoming_step)
        return SearchResult(
            config=deepcopy(node.config),
            cost=cost,
            quality=quality,
            origin=origin,
            trace=trace,
            generation=generation,
            parent_id=node.parent_id,
        )

    def _seed_root_model_baselines(
        self,
        root_node: SearchTreeNode,
        frontier: ParetoFrontier,
        candidates: List[SearchResult],
    ) -> None:
        """Seed frontier with root-level model-swap baseline candidates."""
        if self._evaluated_count >= self._moar_config.max_evaluations:
            return

        baseline_seeds = self._build_root_model_baselines(root_node)
        for baseline_node in baseline_seeds:
            if self._evaluated_count >= self._moar_config.max_evaluations:
                break

            origin = baseline_node.incoming_action_key or "baseline:model_swap"
            result = self._evaluate_node(baseline_node, origin=origin, generation=0)
            candidates.append(result)
            frontier_update = frontier.add_with_details(
                quality=result.quality,
                total_cost=self._total_cost(result),
                candidate_id=baseline_node.node_id,
                payload=result,
            )
            self._backpropagate([root_node, baseline_node], frontier_update.reward)

    def _build_root_model_baselines(self, root_node: SearchTreeNode) -> List[SearchTreeNode]:
        """Create root-level baseline nodes by swapping all LLM models at once."""
        process = root_node.config.get("process", [])
        if not isinstance(process, list) or not process:
            return []

        current_models: List[str] = []
        for identity in root_node.operators:
            model_value = self._extract_model_value(identity.params)
            if model_value:
                current_models.append(model_value)

        if not current_models:
            return []

        dominant_model = current_models[0]
        candidate_models = self._model_registry.get_swap_candidates(dominant_model)
        baseline_nodes: List[SearchTreeNode] = []

        for to_model in candidate_models:
            updated_config, replaced = self._swap_models_in_config(root_node.config, to_model)
            if replaced == 0:
                continue

            before_model = dominant_model
            action_key = f"baseline:swap_model:{to_model}"
            baseline_node = SearchTreeNode.from_config(
                updated_config,
                parent_id=root_node.node_id,
                node_id=self._next_node_id(),
                depth=0,
            )
            baseline_node.incoming_action_key = action_key
            baseline_node.incoming_directive_result = DirectiveResult(
                ok=True,
                applied=True,
                directive_name="swap_model",
                message=f"root baseline {before_model} -> {to_model}",
                config_before=deepcopy(root_node.config),
                config_after=deepcopy(updated_config),
                details={
                    "action_type": "model_swap",
                    "action_scope": "root_baseline",
                    "from_model": before_model,
                    "to_model": to_model,
                    "replaced_operators": replaced,
                },
            )
            baseline_nodes.append(baseline_node)

        return baseline_nodes

    def _extract_model_value(self, params: Dict[str, Any]) -> str:
        """Extract model value from operator params."""
        for key in ("api_model", "model"):
            value = params.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    def _swap_models_in_config(self, config: DJExecutableConfig, to_model: str) -> Tuple[DJExecutableConfig, int]:
        """Swap compatible LLM operator models in a config copy."""
        updated = deepcopy(config)
        process = updated.get("process", [])
        if not isinstance(process, list):
            return updated, 0

        replaced = 0
        index = ProcessIndex.build(process)
        for idx, identity in enumerate(index.identities):
            current_model = self._extract_model_value(identity.params)
            if not current_model:
                continue
            if not self._model_registry.is_swap_compatible(current_model, to_model):
                continue

            step = process[idx]
            if not isinstance(step, dict) or len(step) != 1:
                continue
            op_name = next(iter(step.keys()))
            params = step.get(op_name, {})
            if not isinstance(params, dict):
                continue

            key = "api_model" if isinstance(params.get("api_model"), str) else "model"
            if not isinstance(params.get(key), str):
                continue

            params[key] = to_model
            process[idx] = {op_name: params}
            replaced += 1

        return updated, replaced

    def _total_cost(self, result: SearchResult) -> float:
        """Combine cost dimensions into a scalar for Pareto utilities."""
        return result.cost.llm_token_cost + result.cost.wall_time_sec

    def _select_path(self, root: SearchTreeNode) -> List[SearchTreeNode]:
        """Select a path by descending UCB while nodes are fully expanded."""
        path = [root]
        node = root

        while node.depth < self._moar_config.max_tree_depth:
            if self._has_untried_actions(node):
                break
            if not node.children:
                break
            node = self._select_child_ucb(node)
            path.append(node)

        return path

    def _has_untried_actions(self, node: SearchTreeNode) -> bool:
        """Check whether a node still has available untried actions."""
        for action in self._generate_actions_for_node(node):
            if not node.is_action_used(action.action_key):
                return True
        return False

    def _select_child_ucb(self, node: SearchTreeNode) -> SearchTreeNode:
        """Select child with highest UCB score (deterministic tie-break)."""
        best_key = ""
        best_score = float("-inf")

        parent_visits = max(node.visit_count, 1)
        for action_key in sorted(node.children.keys()):
            child = node.children[action_key]
            if child.visit_count == 0:
                score = float("inf")
            else:
                exploration = self._moar_config.exploration_weight * math.sqrt(
                    math.log(float(parent_visits)) / float(child.visit_count)
                )
                score = child.mean_reward() + exploration

            if score > best_score or (score == best_score and action_key < best_key):
                best_score = score
                best_key = action_key

        return node.children[best_key]

    def _expand(self, node: SearchTreeNode) -> Optional[SearchTreeNode]:
        """Expand one untried action from a selected node."""
        if node.depth >= self._moar_config.max_tree_depth:
            return None

        actions = [
            action
            for action in self._generate_actions_for_node(node)
            if not node.is_action_used(action.action_key)
        ]
        if not actions:
            return None

        action = self._choose_action(actions)
        node.mark_action_used(action.action_key)

        apply_result = self._apply_action(node.config, action)
        if not apply_result.ok or not apply_result.applied or apply_result.config_after is None:
            return None

        child = SearchTreeNode.from_config(
            apply_result.config_after,
            parent_id=node.node_id,
            node_id=self._next_node_id(),
            depth=node.depth + 1,
        )
        child.incoming_action_key = action.action_key
        node.children[action.action_key] = child
        return child

    def _choose_action(self, actions: List[Any]) -> Any:
        """Choose one action deterministically under a fixed seed."""
        ordered = sorted(actions, key=lambda candidate: candidate.action_key)
        if len(ordered) == 1:
            return ordered[0]
        return ordered[self._rng.randrange(len(ordered))]

    def _generate_actions_for_node(self, node: SearchTreeNode) -> List[Any]:
        """Generate dynamic action list from the node's current configuration."""
        action_space = self._action_builder.build(node.config)
        actions = list(action_space)
        actions.sort(key=lambda action: action.action_key)
        return actions

    def _apply_action(self, config: DJExecutableConfig, action: Any) -> DirectiveResult:
        """Apply one action to a config copy."""
        return action.apply(deepcopy(config))

    def _simulate(self, node: SearchTreeNode) -> SearchResult:
        """Evaluate the expanded node (single-step rollout)."""
        origin = node.incoming_action_key or "expanded"
        return self._evaluate_node(node, origin=origin, generation=node.depth)

    def _backpropagate(self, path: List[SearchTreeNode], reward: float) -> None:
        """Backpropagate frontier-based reward over selected path."""
        for node in path:
            node.visit_count += 1
            node.total_reward += reward


__all__ = ["MOARSearchConfig", "MOARSearchStrategy"]
