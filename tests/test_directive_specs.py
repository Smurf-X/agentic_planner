# -*- coding: utf-8 -*-
"""Tests for directive specs and instantiated directives."""

from __future__ import annotations

from agentic_planner.optimizer.action import ActionSpaceBuilder
from agentic_planner.optimizer.directives.instances import InstantiatedDirective
from agentic_planner.optimizer.directives.registry import (
    get_directive_spec,
    list_directive_spec_names,
)
from agentic_planner.optimizer.op_locator import ProcessIndex, TargetLocator


def _sample_config():
    return {
        "dataset_path": "data.jsonl",
        "export_path": "output.jsonl",
        "process": [
            {"text_length_filter": {"min_len": 10, "max_len": 1000}},
            {"text_length_filter": {"min_len": 20, "max_len": 500}},
            {"language_id_score_filter": {"min_score": 0.5}},
        ],
    }


def test_first_wave_directive_specs_registered():
    names = list_directive_spec_names()

    assert "tighten_threshold" in names
    assert "loosen_threshold" in names
    assert "remove_redundant_op" in names
    assert "safe_reorder_local" in names


def test_directive_specs_include_safety_and_applicability_metadata():
    spec = get_directive_spec("tighten_threshold")

    assert spec is not None
    assert spec.safety_level == "safe"
    assert spec.applicability is not None
    assert spec.applicability.per_operator is True
    assert "text_length_filter" in spec.applicability.applicable_op_types
    assert spec.applicability.allows_operator("text_length_filter") is True
    assert spec.applicability.allows_operator("generate_qa_from_text_mapper") is False
    assert spec.is_search_safe() is True


def test_instantiated_directive_signature_is_replayable_and_stable():
    spec = get_directive_spec("tighten_threshold")
    assert spec is not None

    locator = TargetLocator(operator_id="op-abc", audit_identity_hash="h-1")
    inst1 = spec.instantiate(params={"intensity": 0.2}, target_locator=locator)
    inst2 = spec.instantiate(params={"intensity": 0.2}, target_locator=locator)
    inst3 = spec.instantiate(params={"intensity": 0.5}, target_locator=locator)

    assert isinstance(inst1, InstantiatedDirective)
    assert inst1.replay_signature == inst2.replay_signature
    assert inst1.replay_signature != inst3.replay_signature


def test_instantiated_directive_target_locator_support():
    spec = get_directive_spec("tighten_threshold")
    assert spec is not None

    cfg = _sample_config()
    index = ProcessIndex.build(cfg["process"])
    locator = index.identities[0].to_target_locator()

    inst = spec.instantiate(params={"intensity": 0.5}, target_locator=locator)
    result = inst.apply(cfg)

    assert result.ok is True
    assert result.applied is True
    assert result.config_after is not None

    process = result.config_after["process"]
    first_min = process[0]["text_length_filter"]["min_len"]
    second_min = process[1]["text_length_filter"]["min_len"]
    assert first_min > 10
    assert second_min == 20


def test_action_space_builder_applies_spec_metadata_rules():
    cfg = _sample_config()
    builder = ActionSpaceBuilder(
        directive_names=["tighten_threshold", "safe_reorder_local"],
    )

    space = builder.build(cfg)

    assert space.operator_count == 3
    assert len(space.actions) == 3
    assert all(action.directive_name == "tighten_filters" for action in space.actions)


def test_instantiated_directive_fails_closed_for_missing_target_locator():
    spec = get_directive_spec("tighten_threshold")
    assert spec is not None

    cfg = _sample_config()
    missing = TargetLocator(operator_id="op-missing", audit_identity_hash="missing")
    inst = spec.instantiate(params={"intensity": 0.2}, target_locator=missing)
    result = inst.apply(cfg)

    assert result.ok is False
    assert result.applied is False
    assert "no longer exists" in result.message
