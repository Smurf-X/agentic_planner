# -*- coding: utf-8 -*-
"""Tests for Action-based BeamSearch."""

import pytest
from unittest.mock import Mock, MagicMock

from agentic_planner.optimizer.search.beam import (
    BeamSearchConfig,
    BeamSearchStrategy,
)
from agentic_planner.optimizer.action import ActionSpaceBuilder
from agentic_planner.optimizer.directives.adjust_threshold import (
    TightenFiltersDirective,
    LoosenFiltersDirective,
)
from agentic_planner.contracts.recipe import DJExecutableConfig
from agentic_planner.contracts.cost import CostBreakdown


@pytest.fixture
def sample_config() -> DJExecutableConfig:
    return {
        "dataset_path": "data.jsonl",
        "export_path": "output.jsonl",
        "process": [
            {"text_length_filter": {"min_len": 10, "max_len": 1000}},
            {"words_num_filter": {"min_num": 5, "max_num": 100}},
        ],
    }


@pytest.fixture
def mock_evaluator():
    evaluator = Mock()
    evaluator.evaluate = Mock(
        return_value=(CostBreakdown(llm_token_cost=0.1, wall_time_sec=1.0), 0.7)
    )
    return evaluator


class TestBeamSearchConfig:
    def test_default_config(self):
        config = BeamSearchConfig()

        assert config.beam_width == 4
        assert config.max_iterations == 3
        assert config.use_llm_selection == False
        assert config.track_pareto == True

    def test_custom_config(self):
        config = BeamSearchConfig(
            beam_width=8,
            max_iterations=5,
            use_llm_selection=True,
            optimize_goal="cost",
        )

        assert config.beam_width == 8
        assert config.max_iterations == 5
        assert config.use_llm_selection == True
        assert config.optimize_goal == "cost"


class TestBeamSearchStrategy:
    def test_search_without_llm(self, sample_config, mock_evaluator):
        action_builder = ActionSpaceBuilder(directives=[TightenFiltersDirective(intensity=0.1)])

        config = BeamSearchConfig(
            beam_width=2,
            max_iterations=2,
            use_llm_selection=False,
        )

        strategy = BeamSearchStrategy(
            config=config,
            evaluator=mock_evaluator,
            action_builder=action_builder,
        )

        report = strategy.search(sample_config)

        assert report.ok == True
        assert len(report.candidates) > 0
        assert report.total_evaluations > 0

    def test_search_tracks_pareto(self, sample_config, mock_evaluator):
        action_builder = ActionSpaceBuilder(
            directives=[
                TightenFiltersDirective(intensity=0.1),
                LoosenFiltersDirective(intensity=0.1),
            ]
        )

        config = BeamSearchConfig(
            beam_width=2,
            max_iterations=2,
            track_pareto=True,
        )

        strategy = BeamSearchStrategy(
            config=config,
            evaluator=mock_evaluator,
            action_builder=action_builder,
        )

        report = strategy.search(sample_config)

        assert report.ok == True
        assert len(report.pareto_front) >= 1

    def test_action_space_size_in_metrics(self, sample_config, mock_evaluator):
        action_builder = ActionSpaceBuilder(directives=[TightenFiltersDirective()])

        config = BeamSearchConfig(beam_width=2, max_iterations=1)

        strategy = BeamSearchStrategy(
            config=config,
            evaluator=mock_evaluator,
            action_builder=action_builder,
        )

        report = strategy.search(sample_config)

        assert "action_space_size" in report.metrics


class TestActionBasedOptimization:
    def test_action_targets_specific_operator(self, sample_config, mock_evaluator):
        action_builder = ActionSpaceBuilder(directives=[TightenFiltersDirective(intensity=0.2)])

        config = BeamSearchConfig(
            beam_width=4,
            max_iterations=1,
        )

        strategy = BeamSearchStrategy(
            config=config,
            evaluator=mock_evaluator,
            action_builder=action_builder,
        )

        report = strategy.search(sample_config)

        for candidate in report.candidates:
            if candidate.origin != "root":
                assert "+" in candidate.origin or candidate.generation > 0

    def test_used_actions_prevents_repetition(self, sample_config, mock_evaluator):
        action_builder = ActionSpaceBuilder(directives=[TightenFiltersDirective(intensity=0.1)])

        config = BeamSearchConfig(
            beam_width=2,
            max_iterations=3,
        )

        strategy = BeamSearchStrategy(
            config=config,
            evaluator=mock_evaluator,
            action_builder=action_builder,
        )

        report = strategy.search(sample_config)

        max_generation = max(c.generation for c in report.candidates)
        assert max_generation <= config.max_iterations
