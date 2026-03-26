# -*- coding: utf-8 -*-
"""Tests for directive specs and instantiated directives."""

from __future__ import annotations

import importlib

from agentic_planner.optimizer.action import ActionSpaceBuilder
from agentic_planner.optimizer.directive_inference import DirectiveInferenceEngine
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


def _sample_llm_config():
    return {
        "dataset_path": "data.jsonl",
        "export_path": "output.jsonl",
        "process": [
            {
                "llm_filter": {
                    "prompt": "Keep only high quality records.",
                    "api_model": "gpt-4o-mini",
                }
            }
        ],
    }


def test_first_wave_directive_specs_registered():
    names = list_directive_spec_names()

    assert "tighten_threshold" in names
    assert "loosen_threshold" in names
    assert "remove_redundant_op" in names
    assert "safe_reorder_local" in names


def test_second_wave_directive_specs_registered():
    names = list_directive_spec_names()

    assert "swap_model" in names
    assert "rewrite_prompt" in names
    assert "add_few_shot_examples" in names


def test_gleaning_directive_module_removed() -> None:
    try:
        importlib.import_module("agentic_planner.optimizer.directives.gleaning")
    except ModuleNotFoundError:
        assert True
    else:
        assert False, "gleaning directive module should not be importable"


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
    assert all(action.directive_template == "tighten_threshold" for action in space.actions)
    assert all(action.instantiate_params["intensity"] == 0.1 for action in space.actions)


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


def test_directive_inference_heuristics_emit_template_target_plan() -> None:
    cfg = _sample_config()
    index = ProcessIndex.build(cfg["process"])

    engine = DirectiveInferenceEngine(llm_client=None)
    recommendations = engine.analyze(cfg)

    assert recommendations.recommendations
    first = recommendations.recommendations[0]
    assert first.directive_template
    if first.target_locator is not None:
        assert first.target_locator.operator_id in {
            identity.operator_id for identity in index.identities
        }


def test_swap_model_spec_applies_and_emits_model_trace_fields() -> None:
    cfg = _sample_llm_config()
    index = ProcessIndex.build(cfg["process"])
    locator = index.identities[0].to_target_locator()

    spec = get_directive_spec("swap_model")
    assert spec is not None

    inst = spec.instantiate(
        params={
            "from_model": "gpt-4o-mini",
            "to_model": "gpt-4o",
        },
        target_locator=locator,
    )
    result = inst.apply(cfg)

    assert result.ok is True
    assert result.applied is True
    assert result.details["action_type"] == "model_swap"
    assert result.details["from_model"] == "gpt-4o-mini"
    assert result.details["to_model"] == "gpt-4o"
    assert result.details["model_param_key"] == "api_model"


def test_rewrite_prompt_spec_emits_prompt_trace_fields() -> None:
    cfg = _sample_llm_config()
    index = ProcessIndex.build(cfg["process"])
    locator = index.identities[0].to_target_locator()

    spec = get_directive_spec("rewrite_prompt")
    assert spec is not None

    inst = spec.instantiate(
        params={
            "new_prompt": "Keep concise and informative records only.",
        },
        target_locator=locator,
    )
    result = inst.apply(cfg)

    assert result.ok is True
    assert result.applied is True
    assert result.details["action_type"] == "prompt_rewrite"
    assert result.details["prompt_before_chars"] > 0
    assert result.details["prompt_after_chars"] > 0
    assert result.details["prompt_key"] == "prompt"


def test_add_few_shot_examples_spec_emits_examples_trace_fields() -> None:
    cfg = _sample_llm_config()
    index = ProcessIndex.build(cfg["process"])
    locator = index.identities[0].to_target_locator()

    spec = get_directive_spec("add_few_shot_examples")
    assert spec is not None

    inst = spec.instantiate(
        params={
            "examples": [
                {"input": "short lorem", "output": "drop"},
                {"input": "well formed answer", "output": "keep"},
            ],
        },
        target_locator=locator,
    )
    result = inst.apply(cfg)

    assert result.ok is True
    assert result.applied is True
    assert result.details["action_type"] == "few_shot_examples"
    assert result.details["examples_added"] == 2
    assert result.details["prompt_key"] == "prompt"
