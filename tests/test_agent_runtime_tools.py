# -*- coding: utf-8 -*-
"""Tests for runtime tool wrappers and router actions."""

from __future__ import annotations

from pathlib import Path

from agent_runtime.api.schemas import ToolResponse
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

    assert isinstance(result, ToolResponse)
    assert result.ok is True
    assert result.error is None
    assert result.data["yaml_text"] == "process:\n  - clean_text_mapper: {}\n"
    assert result.data["intent"] == "clean text"


def test_optimize_yaml_tool_returns_structured_payload() -> None:
    """Optimize tool should return normalized success envelope."""
    result = optimize_yaml_tool(
        yaml_text_or_path="process:\n  - clean_text_mapper: {}\n",
        objective="cost",
        model_config_path="/tmp/models.yaml",
        options={"max_iterations": 1},
    )

    assert isinstance(result, ToolResponse)
    assert result.ok is True
    assert result.error is None
    assert result.data["optimized_yaml"] == "process:\n  - clean_text_mapper: {}\n"
    assert result.data["objective"] == "cost"


def test_optimize_yaml_tool_reads_yaml_from_file_path(tmp_path: Path) -> None:
    """Optimize tool should read YAML from disk path."""
    yaml_path = tmp_path / "pipeline.yaml"
    yaml_path.write_text("process:\n  - clean_text_mapper: {}\n", encoding="utf-8")

    result = optimize_yaml_tool(
        yaml_text_or_path=str(yaml_path),
        objective="quality",
        model_config_path="/tmp/models.yaml",
        options={},
    )

    assert result.ok is True
    assert result.error is None
    assert result.data["optimized_yaml"] == "process:\n  - clean_text_mapper: {}\n"


def test_tool_missing_required_arg_returns_error_envelope() -> None:
    """Missing required args should return failure envelope with error."""
    result = generate_yaml_tool(
        intent="x",
        dataset_path="/tmp/a.jsonl",
        model_config_path="",
        options={},
    )

    assert isinstance(result, ToolResponse)
    assert result.ok is False
    assert result.error == "missing required argument: model_config_path"


def test_metadata_tools_return_normalized_envelopes() -> None:
    """List/explain/validate wrappers should return normalized envelopes."""
    list_result = list_ops_tool(options={})
    explain_result = explain_op_tool(operator_name="clean_text_mapper", options={})
    validate_result = validate_yaml_tool(yaml_text_or_path="process: []", options={})

    for result in [list_result, explain_result, validate_result]:
        assert isinstance(result, ToolResponse)
        assert isinstance(result.data, dict)
        if result.ok:
            assert result.error is None
        else:
            assert result.error

    assert list_result.data["operators"][0]["name"] == "clean_text_mapper"
    assert explain_result.data["name"] == "clean_text_mapper"
    assert validate_result.data["valid"] is False


def test_validate_yaml_tool_returns_error_on_contract_validation_failure() -> None:
    """Validation failures should include a non-null error message."""
    result = validate_yaml_tool(yaml_text_or_path="process: []", options={})

    assert result.ok is False
    assert result.error == "validation failed"
    assert result.data["valid"] is False


def test_validate_yaml_tool_reads_yaml_from_file_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Validate tool should support YAML file path input."""
    yaml_path = tmp_path / "valid.yaml"
    yaml_path.write_text("process:\n  - clean_text_mapper: {}\n", encoding="utf-8")

    monkeypatch.setattr(
        "agent_runtime.tools.validate_yaml.validate_executable_config",
        lambda cfg: [],
    )

    result = validate_yaml_tool(yaml_text_or_path=str(yaml_path), options={"strict": True})

    assert result.ok is True
    assert result.error is None
    assert result.data["valid"] is True
    assert result.data["errors"] == []


def test_optimize_yaml_tool_handles_file_read_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Optimize tool should return normalized error on read failures."""
    yaml_path = tmp_path / "input.yaml"
    yaml_path.write_text("process:\n  - clean_text_mapper: {}\n", encoding="utf-8")

    def _raise_oserror(self: Path, encoding: str = "utf-8") -> str:  # noqa: ARG001
        raise OSError("mock read failure")

    monkeypatch.setattr(Path, "read_text", _raise_oserror)

    result = optimize_yaml_tool(
        yaml_text_or_path=str(yaml_path),
        objective="quality",
        model_config_path="/tmp/models.yaml",
        options={},
    )

    assert result.ok is False
    assert result.error == "failed to read yaml input"


def test_validate_yaml_tool_handles_file_read_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Validate tool should return normalized error on read failures."""
    yaml_path = tmp_path / "input.yaml"
    yaml_path.write_text("process:\n  - clean_text_mapper: {}\n", encoding="utf-8")

    def _raise_oserror(self: Path, encoding: str = "utf-8") -> str:  # noqa: ARG001
        raise OSError("mock read failure")

    monkeypatch.setattr(Path, "read_text", _raise_oserror)

    result = validate_yaml_tool(yaml_text_or_path=str(yaml_path), options={})

    assert result.ok is False
    assert result.error == "failed to read yaml input"


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

    assert result.ok is True
    assert result.error is None
    assert result.data["intent"] == "clean text"


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

    assert result.ok is True
    assert result.error is None
    assert result.data["objective"] == "quality"


def test_router_routes_list_action_to_tool() -> None:
    """Router should route list action to list tool."""
    router = Router()

    result = router.route(action="list", payload={"options": {}})

    assert result.ok is True
    assert result.error is None
    assert result.data["operators"]


def test_router_routes_list_ops_alias_to_tool() -> None:
    """Router should route list_ops alias to list tool."""
    router = Router()

    result = router.route(action="list_ops", payload={"options": {}})

    assert result.ok is True
    assert result.error is None
    assert result.data["operators"][0]["name"] == "clean_text_mapper"


def test_router_routes_explain_action_to_tool() -> None:
    """Router should route explain action to explain tool."""
    router = Router()

    result = router.route(
        action="explain",
        payload={"operator_name": "clean_text_mapper", "options": {}},
    )

    assert result.ok is True
    assert result.error is None
    assert result.data["name"] == "clean_text_mapper"


def test_router_routes_explain_op_alias_to_tool() -> None:
    """Router should route explain_op alias to explain tool."""
    router = Router()

    result = router.route(
        action="explain_op",
        payload={"operator_name": "clean_text_mapper", "options": {}},
    )

    assert result.ok is True
    assert result.error is None
    assert result.data["name"] == "clean_text_mapper"


def test_router_routes_validate_action_to_tool() -> None:
    """Router should route validate action to validate tool."""
    router = Router()

    result = router.route(
        action="validate",
        payload={"yaml_text_or_path": "process: []", "options": {}},
    )

    assert result.ok is False
    assert result.error == "validation failed"


def test_router_unknown_action_returns_error_envelope() -> None:
    """Router should return a normalized error for unsupported actions."""
    router = Router()

    result = router.route(action="unknown-action", payload={})

    assert result.ok is False
    assert result.error == "unsupported action: unknown-action"
