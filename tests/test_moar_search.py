# -*- coding: utf-8 -*-
"""Tests for MOAR search entry points."""

from __future__ import annotations

from copy import deepcopy

import pytest

from agentic_planner.contracts.cost import CostBreakdown
from agentic_planner.optimizer.directives.base import DirectiveResult
from agentic_planner.optimizer.optimization_config import OptimizationConfig
from agentic_planner.optimizer.runner import OptimizationRunMode, OptimizationRunner
from agentic_planner.optimizer.search import create_search_strategy
from agentic_planner.optimizer.search.base import SearchReport, SearchResult, SearchStrategyType
from agentic_planner.optimizer.search.moar import MOARSearchConfig, MOARSearchStrategy
from agentic_planner.optimizer.search.tree_node import SearchTreeNode


def test_default_search_mode_builds_moar_config() -> None:
    """Search modes should auto-fill MOAR config defaults."""
    cfg = OptimizationConfig(run_mode="search_only")

    assert isinstance(cfg.search, MOARSearchConfig)
    assert cfg.search.strategy == SearchStrategyType.MCTS
    assert cfg.search_execution_boundary.to_runtime_overrides() == {
        "op_fusion": False,
        "checkpoint_optimization": False,
        "partition_optimization": False,
    }


def test_runner_uses_moar_search_strategy(monkeypatch) -> None:
    """Runner should instantiate MOAR strategy for search stages."""
    calls = {"constructed": 0, "searched": 0}
    captured_root = {"value": None}

    class StubMOARSearchStrategy:
        def __init__(self, config, evaluator=None):
            _ = evaluator
            assert isinstance(config, MOARSearchConfig)
            calls["constructed"] += 1

        def search(self, root):
            captured_root["value"] = root
            calls["searched"] += 1
            return SearchReport(
                ok=True,
                candidates=[
                    SearchResult(
                        config={"process": []},
                        cost=CostBreakdown(),
                        quality=0.8,
                        origin="stub",
                    )
                ],
            )

    monkeypatch.setattr(
        "agentic_planner.optimizer.runner.MOARSearchStrategy",
        StubMOARSearchStrategy,
    )

    runner = OptimizationRunner(
        mode=OptimizationRunMode.SEARCH_ONLY,
        moar_config={},
    )
    result = runner.run({"process": []})

    assert calls["constructed"] == 1
    assert calls["searched"] == 1
    assert result.best_quality == 0.8
    assert captured_root["value"]["op_fusion"] is False
    assert captured_root["value"]["checkpoint_optimization"] is False
    assert captured_root["value"]["partition_optimization"] is False


def test_moar_strategy_constructible_from_defaults() -> None:
    """MOAR strategy should be constructible with default config."""
    strategy = MOARSearchStrategy(MOARSearchConfig())

    report = strategy.search({"process": []})
    assert report.ok is True
    assert len(report.candidates) == 1
    assert report.candidates[0].origin == "root"


def test_factory_accepts_mcts_and_moar_aliases() -> None:
    """Search factory should support both strategy aliases."""
    strategy_mcts = create_search_strategy("mcts", {}, evaluator=None)
    strategy_moar = create_search_strategy("moar", {}, evaluator=None)

    assert isinstance(strategy_mcts, MOARSearchStrategy)
    assert isinstance(strategy_moar, MOARSearchStrategy)


def test_runner_propagates_failed_search_report(monkeypatch) -> None:
    """Runner should surface failed search report errors."""

    class FailingMOARSearchStrategy:
        def __init__(self, config, evaluator=None):
            _ = config
            _ = evaluator

        def search(self, root):
            _ = root
            return SearchReport(ok=False, candidates=[], errors=["search failed"])

    monkeypatch.setattr(
        "agentic_planner.optimizer.runner.MOARSearchStrategy",
        FailingMOARSearchStrategy,
    )

    runner = OptimizationRunner(
        mode=OptimizationRunMode.SEARCH_ONLY,
        moar_config={},
    )
    result = runner.run({"process": []})

    assert result.ok is False
    assert result.errors == ["search failed"]


def test_moar_config_rejects_non_mcts_strategy() -> None:
    """MOAR config should reject non-MCTS strategy values."""
    with pytest.raises(ValueError):
        MOARSearchConfig(strategy=SearchStrategyType.RANDOM)


def test_runner_accepts_search_config_alias(monkeypatch) -> None:
    """Runner should accept legacy search_config alias."""
    captured = {"max_iterations": None}

    class StubMOARSearchStrategy:
        def __init__(self, config, evaluator=None):
            _ = evaluator
            captured["max_iterations"] = config.max_iterations

        def search(self, root):
            _ = root
            return SearchReport(ok=True, candidates=[])

    monkeypatch.setattr(
        "agentic_planner.optimizer.runner.MOARSearchStrategy",
        StubMOARSearchStrategy,
    )

    runner = OptimizationRunner(
        mode=OptimizationRunMode.SEARCH_ONLY,
        search_config={"max_iterations": 7},
    )
    runner.run({"process": []})

    assert captured["max_iterations"] == 7


class _FakeAction:
    def __init__(self, action_key: str, marker: str = "") -> None:
        self._action_key = action_key
        self._marker = marker

    @property
    def action_key(self) -> str:
        return self._action_key

    def apply(self, config):
        after = deepcopy(config)
        if self._marker:
            after[self._marker] = True
        return DirectiveResult(
            ok=True,
            applied=True,
            directive_name="fake",
            config_before=config,
            config_after=after,
        )


class _MarkerEvaluator:
    def evaluate(self, config):
        if config.get("dominated"):
            return CostBreakdown(llm_token_cost=1.0, wall_time_sec=1.0), 0.1
        return CostBreakdown(llm_token_cost=0.0, wall_time_sec=0.0), 0.5


def test_tree_node_action_memory_uses_canonical_action_keys() -> None:
    """Used-action memory should dedupe by action_key, not object identity."""
    node = SearchTreeNode.from_config({"process": []})
    action_a1 = _FakeAction("same:key")
    action_a2 = _FakeAction("same:key")
    action_b = _FakeAction("other:key")

    node.mark_action_used(action_a1.action_key)

    assert node.is_action_used(action_a1.action_key) is True
    assert node.is_action_used(action_a2.action_key) is True
    assert node.is_action_used(action_b.action_key) is False


def test_moar_search_generates_actions_per_node_dynamically(monkeypatch) -> None:
    """Action generation should run for selected child nodes, not only root."""
    strategy = MOARSearchStrategy(MOARSearchConfig(max_iterations=3, seed=3))
    visited_depths = []

    def fake_generate(node):
        visited_depths.append(node.depth)
        if node.depth == 0:
            return [_FakeAction("root:action")]
        if node.depth == 1:
            return [_FakeAction("child:action")]
        return []

    monkeypatch.setattr(strategy, "_generate_actions_for_node", fake_generate)

    report = strategy.search({"process": []})

    assert report.ok is True
    assert len(report.candidates) >= 3
    assert 0 in visited_depths
    assert 1 in visited_depths


def test_moar_search_early_stops_when_frontier_stalls(monkeypatch) -> None:
    """Search should stop once frontier reward stays below threshold for patience window."""
    strategy = MOARSearchStrategy(
        MOARSearchConfig(
            max_iterations=12,
            early_stop_patience=2,
            frontier_improvement_threshold=1.0,
            seed=11,
        ),
        evaluator=_MarkerEvaluator(),
    )

    def fake_generate(node):
        if node.depth >= 1:
            return []
        return [
            _FakeAction("dominated:action:1", marker="dominated"),
            _FakeAction("dominated:action:2", marker="dominated"),
        ]

    monkeypatch.setattr(strategy, "_generate_actions_for_node", fake_generate)

    report = strategy.search({"process": []})

    assert report.metrics["early_stopped"] is True
    assert report.metrics["stop_reason"] == "frontier_improvement_stalled"
    assert report.total_iterations < 12
