# -*- coding: utf-8 -*-
"""Tests for Action and ActionSpace."""

import pytest

from agentic_planner.optimizer.action import Action, ActionSpace, ActionSpaceBuilder
from agentic_planner.optimizer.directives.adjust_threshold import (
    TightenFiltersDirective,
    LoosenFiltersDirective,
)
from agentic_planner.contracts.recipe import DJExecutableConfig


@pytest.fixture
def sample_config() -> DJExecutableConfig:
    return {
        "dataset_path": "data.jsonl",
        "export_path": "output.jsonl",
        "process": [
            {"text_length_filter": {"min_len": 10, "max_len": 1000}},
            {"words_num_filter": {"min_num": 5, "max_num": 100}},
            {"perplexity_filter": {"max_ppl": 500}},
        ],
    }


class TestAction:
    def test_action_creation(self):
        directive = TightenFiltersDirective(intensity=0.1)
        action = Action(
            operator_index=0,
            operator_name="text_length_filter",
            directive=directive,
        )

        assert action.operator_index == 0
        assert action.operator_name == "text_length_filter"
        assert action.directive_name == "tighten_filters"

    def test_action_hash_and_equality(self):
        directive = TightenFiltersDirective(intensity=0.1)
        action1 = Action(
            operator_index=0,
            operator_name="text_length_filter",
            directive=directive,
        )
        action2 = Action(
            operator_index=0,
            operator_name="text_length_filter",
            directive=directive,
        )
        action3 = Action(
            operator_index=1,
            operator_name="text_length_filter",
            directive=directive,
        )

        assert action1 == action2
        assert action1 != action3
        assert hash(action1) == hash(action2)


class TestActionSpace:
    def test_action_space_creation(self):
        directive = TightenFiltersDirective()
        actions = [
            Action(0, "text_length_filter", directive),
            Action(1, "words_num_filter", directive),
        ]

        space = ActionSpace(actions=actions, config_hash="abc123", operator_count=2)

        assert len(space) == 2
        assert space[0].operator_index == 0
        assert space[1].operator_index == 1

    def test_action_space_filter(self):
        directive1 = TightenFiltersDirective()
        directive2 = LoosenFiltersDirective()
        actions = [
            Action(0, "text_length_filter", directive1),
            Action(0, "text_length_filter", directive2),
            Action(1, "words_num_filter", directive1),
        ]

        space = ActionSpace(actions=actions)

        filtered = space.filter(lambda a: a.directive_name == "tighten_filters")
        assert len(filtered) == 2

        filtered2 = space.get_for_operator(0)
        assert len(filtered2) == 2

    def test_action_space_exclude(self):
        directive = TightenFiltersDirective()
        actions = [
            Action(0, "text_length_filter", directive),
            Action(1, "words_num_filter", directive),
        ]

        space = ActionSpace(actions=actions)

        used = {(0, "tighten_filters")}
        remaining = space.exclude(used)

        assert len(remaining) == 1
        assert remaining[0].operator_index == 1


class TestActionSpaceBuilder:
    def test_build_action_space(self, sample_config):
        builder = ActionSpaceBuilder(
            directives=[TightenFiltersDirective(), LoosenFiltersDirective()]
        )

        space = builder.build(sample_config)

        assert space.operator_count == 3
        assert len(space) == 6  # 3 operators * 2 directives

    def test_build_with_applicable_types(self, sample_config):
        builder = ActionSpaceBuilder(directives=[TightenFiltersDirective()])

        space = builder.build(sample_config)

        assert len(space) == 3  # All 3 filters are applicable

    def test_build_for_specific_operator(self, sample_config):
        builder = ActionSpaceBuilder(
            directives=[TightenFiltersDirective(), LoosenFiltersDirective()]
        )

        actions = builder.build_for_operator(sample_config, 0)

        assert len(actions) == 2
        for action in actions:
            assert action.operator_index == 0
