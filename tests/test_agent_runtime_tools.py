# -*- coding: utf-8 -*-
"""Tests for runtime tool wrappers and router actions."""

from __future__ import annotations

from agent_runtime.orchestrator.router import Router
from agent_runtime.tools.explain_op import explain_op_tool
from agent_runtime.tools.generate_yaml import generate_yaml_tool
from agent_runtime.tools.list_ops import list_ops_tool
from agent_runtime.tools.optimize_yaml import optimize_yaml_tool
from agent_runtime.tools.validate_yaml import validate_yaml_tool


def test_generate_yaml_tool_returns_structured_payload() -> None:
    """Generate tool should return normalized success envelope."""
    result = generate_yaml_tool(
        intent="clean text",
        dataset_path="/tmp/a.jsonl",
        model_config_path="/tmp/models.yaml",
        options={"mode": "default"},
    )

    assert set(result.keys()) == {"ok", "data", "timing_ms", "error"}
    assert result["ok"] is True
    assert result["error"] is None
    assert isinstance(result["data"], dict)


def test_optimize_yaml_tool_returns_structured_payload() -> None:
    """Optimize tool should return normalized success envelope."""
    result = optimize_yaml_tool(
        yaml_text_or_path="process:\n  - clean_text_mapper: {}\n",
        objective="cost",
        model_config_path="/tmp/models.yaml",
        options={"max_iterations": 1},
    )

    assert set(result.keys()) == {"ok", "data", "timing_ms", "error"}
    assert result["ok"] is True
    assert result["error"] is None
    assert isinstance(result["data"], dict)


def test_tool_missing_required_arg_returns_error_envelope() -> None:
    """Missing required args should return failure envelope with error."""
    result = generate_yaml_tool(
        intent="x",
        dataset_path="/tmp/a.jsonl",
        model_config_path="",
        options={},
    )

    assert result["ok"] is False
    assert set(result.keys()) == {"ok", "data", "timing_ms", "error"}
    assert "error" in result
    assert result["error"]


def test_metadata_tools_return_normalized_envelopes() -> None:
    """List/explain/validate wrappers should return normalized envelopes."""
    list_result = list_ops_tool(options={})
    explain_result = explain_op_tool(operator_name="clean_text_mapper", options={})
    validate_result = validate_yaml_tool(yaml_text_or_path="process: []", options={})

    for result in [list_result, explain_result, validate_result]:
        assert set(result.keys()) == {"ok", "data", "timing_ms", "error"}
        assert isinstance(result["data"], dict)
        if result["ok"]:
            assert result["error"] is None
        else:
            assert result["error"]


def test_validate_yaml_tool_returns_error_on_contract_validation_failure() -> None:
    """Validation failures should include a non-null error message."""
    result = validate_yaml_tool(yaml_text_or_path="process: []", options={})

    assert set(result.keys()) == {"ok", "data", "timing_ms", "error"}
    assert result["ok"] is False
    assert result["error"]


def test_router_routes_generate_action_to_tool() -> None:
    """Router should invoke generate action through tool wrappers."""
    router = Router()

    result = router.route(
        action="generate",
        payload={
            "intent": "clean text",
            "dataset_path": "/tmp/a.jsonl",
            "model_config_path": "/tmp/models.yaml",
            "options": {"mode": "default"},
        },
    )

    assert result["ok"] is True
    assert result["error"] is None
    assert "data" in result


def test_router_routes_optimize_action_to_tool() -> None:
    """Router should invoke optimize action through tool wrappers."""
    router = Router()

    result = router.route(
        action="optimize",
        payload={
            "yaml_text_or_path": "process:\n  - clean_text_mapper: {}\n",
            "objective": "quality",
            "model_config_path": "/tmp/models.yaml",
            "options": {},
        },
    )

    assert set(result.keys()) == {"ok", "data", "timing_ms", "error"}
    assert result["ok"] is True
    assert result["error"] is None


def test_router_routes_list_action_to_tool() -> None:
    """Router should route list action to list tool."""
    router = Router()

    result = router.route(action="list", payload={"options": {}})

    assert set(result.keys()) == {"ok", "data", "timing_ms", "error"}
    assert result["ok"] is True
    assert result["error"] is None


def test_router_routes_explain_action_to_tool() -> None:
    """Router should route explain action to explain tool."""
    router = Router()

    result = router.route(
        action="explain",
        payload={"operator_name": "clean_text_mapper", "options": {}},
    )

    assert set(result.keys()) == {"ok", "data", "timing_ms", "error"}
    assert result["ok"] is True
    assert result["error"] is None


def test_router_routes_validate_action_to_tool() -> None:
    """Router should route validate action to validate tool."""
    router = Router()

    result = router.route(
        action="validate",
        payload={"yaml_text_or_path": "process: []", "options": {}},
    )

    assert set(result.keys()) == {"ok", "data", "timing_ms", "error"}
    assert result["ok"] is False
    assert result["error"]


def test_router_unknown_action_returns_error_envelope() -> None:
    """Router should return a normalized error for unsupported actions."""
    router = Router()

    result = router.route(action="unknown-action", payload={})

    assert result["ok"] is False
    assert "error" in result
