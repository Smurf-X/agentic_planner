# -*- coding: utf-8 -*-
"""Tests for MOAR search entry points."""

from __future__ import annotations

from agentic_planner.contracts.cost import CostBreakdown
from agentic_planner.optimizer.optimization_config import OptimizationConfig
from agentic_planner.optimizer.runner import OptimizationRunMode, OptimizationRunner
from agentic_planner.optimizer.search.base import SearchReport, SearchResult, SearchStrategyType
from agentic_planner.optimizer.search.moar import MOARSearchConfig, MOARSearchStrategy


def test_default_search_mode_builds_moar_config() -> None:
    """Search modes should auto-fill MOAR config defaults."""
    cfg = OptimizationConfig(run_mode="search_only")

    assert isinstance(cfg.search, MOARSearchConfig)
    assert cfg.search.strategy == SearchStrategyType.MCTS


def test_runner_uses_moar_search_strategy(monkeypatch) -> None:
    """Runner should instantiate MOAR strategy for search stages."""
    calls = {"constructed": 0, "searched": 0}

    class StubMOARSearchStrategy:
        def __init__(self, config, evaluator=None):
            _ = evaluator
            assert isinstance(config, MOARSearchConfig)
            calls["constructed"] += 1

        def search(self, root):
            _ = root
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


def test_moar_strategy_constructible_from_defaults() -> None:
    """MOAR strategy should be constructible with default config."""
    strategy = MOARSearchStrategy(MOARSearchConfig())

    report = strategy.search({"process": []})
    assert report.ok is True
    assert report.candidates == []
