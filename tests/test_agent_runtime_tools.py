# -*- coding: utf-8 -*-
"""Tests for runtime tool wrappers and router actions."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from agent_runtime.api.schemas import ToolResponse
from agent_runtime.orchestrator.router import Router
from agent_runtime.tools.explain_op import explain_op_tool
from agent_runtime.tools.generate_yaml import generate_yaml_tool
from agent_runtime.tools.list_ops import list_ops_tool
from agent_runtime.tools.optimize_yaml import optimize_yaml_tool
from agent_runtime.tools.validate_yaml import validate_yaml_tool


def _install_fake_op_searcher(monkeypatch) -> None:
    """Install a fake data_juicer.tools.op_search.OPSearcher module."""

    class _Record:
        def __init__(
            self,
            name: str,
            op_type: str,
            tags,
            desc: str,
            sig: str,
            param_desc: str,
        ) -> None:
            self.name = name
            self.type = op_type
            self.tags = tags
            self.desc = desc
            self.sig = sig
            self.param_desc = param_desc

    class _FakeOPSearcher:
        def __init__(self, specified_op_list=None, include_formatter: bool = False):
            records = [
                _Record(
                    name="alpha_filter",
                    op_type="filter",
                    tags=["quality", "english"],
                    desc="Filter low-confidence English rows.",
                    sig="(threshold: float = 0.8)",
                    param_desc="threshold: keep rows with score >= threshold",
                ),
                _Record(
                    name="beta_mapper",
                    op_type="mapper",
                    tags=["text"],
                    desc="Normalize text fields.",
                    sig="(text_key: str = 'text')",
                    param_desc="text_key: input text field",
                ),
            ]
            if specified_op_list:
                allow = set(specified_op_list)
                records = [record for record in records if record.name in allow]
            if not include_formatter:
                self.op_records = records
            else:
                self.op_records = records

    fake_module = types.ModuleType("data_juicer.tools.op_search")
    fake_module.OPSearcher = _FakeOPSearcher
    monkeypatch.setitem(sys.modules, "data_juicer.tools.op_search", fake_module)


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
        options={"use_real_generator": True},
    )

    assert isinstance(result, ToolResponse)
    assert result.ok is False
    assert result.error


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

    assert "operators" in list_result.data
    if list_result.ok:
        assert isinstance(list_result.data["operators"], list)
    else:
        assert list_result.error == "op search unavailable: data_juicer not installed"

    if explain_result.ok:
        assert explain_result.data["name"] == "clean_text_mapper"
    else:
        assert explain_result.error in {
            "op search unavailable: data_juicer not installed",
            "operator not found: clean_text_mapper",
        }
    assert validate_result.data["valid"] is False


def test_list_ops_tool_uses_op_searcher_records(monkeypatch) -> None:
    """List tool should source operator rows from OPSearcher records."""
    _install_fake_op_searcher(monkeypatch)

    result = list_ops_tool(options={"source": "test"})

    assert result.ok is True
    assert result.error is None
    assert [row["name"] for row in result.data["operators"]] == ["alpha_filter", "beta_mapper"]
    assert result.data["operators"][0]["summary"] == "Filter low-confidence English rows."


def test_explain_op_tool_uses_op_searcher_details(monkeypatch) -> None:
    """Explain tool should return details from OPSearcher metadata."""
    _install_fake_op_searcher(monkeypatch)

    result = explain_op_tool(operator_name="alpha_filter", options={})

    assert result.ok is True
    assert result.error is None
    assert result.data["name"] == "alpha_filter"
    assert result.data["summary"] == "Filter low-confidence English rows."
    assert result.data["signature"] == "(threshold: float = 0.8)"
    assert "threshold" in result.data["param_desc"]


def test_explain_op_tool_returns_not_found_for_unknown_operator(monkeypatch) -> None:
    """Explain tool should return a normalized not-found error."""
    _install_fake_op_searcher(monkeypatch)

    result = explain_op_tool(operator_name="missing_op", options={})

    assert result.ok is False
    assert result.error == "operator not found: missing_op"


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

    if result.ok:
        assert result.error is None
        assert result.data["operators"]
    else:
        assert result.error == "op search unavailable: data_juicer not installed"


def test_router_routes_explain_action_to_tool() -> None:
    """Router should route explain action to explain tool."""
    router = Router()

    result = router.route(
        action="explain",
        payload={"operator_name": "clean_text_mapper", "options": {}},
    )

    if result.ok:
        assert result.error is None
        assert result.data["name"] == "clean_text_mapper"
    else:
        assert result.error in {
            "op search unavailable: data_juicer not installed",
            "operator not found: clean_text_mapper",
        }


def test_router_routes_explain_op_alias_to_tool() -> None:
    """Router should route explain_op alias to explain tool."""
    router = Router()

    result = router.route(
        action="explain_op",
        payload={"operator_name": "clean_text_mapper", "options": {}},
    )

    if result.ok:
        assert result.error is None
        assert result.data["name"] == "clean_text_mapper"
    else:
        assert result.error in {
            "op search unavailable: data_juicer not installed",
            "operator not found: clean_text_mapper",
        }


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


def test_router_returns_normalized_error_for_invalid_payload_and_options() -> None:
    """Router should normalize non-mapping payload/options errors."""
    router = Router()

    none_payload_result = router.route(action="generate", payload=None)
    bad_options_result = router.route(
        action="list",
        payload={"options": "not-a-mapping"},
    )

    assert none_payload_result.ok is False
    assert none_payload_result.error == "invalid payload: expected mapping"
    assert none_payload_result.data["payload_type"] == "NoneType"

    assert bad_options_result.ok is True
    assert bad_options_result.error is None
    assert bad_options_result.data["operators"]


def test_generate_and_optimize_tools_handle_none_options() -> None:
    """Generate/optimize tools should safely normalize None options."""
    generate_result = generate_yaml_tool(
        intent="clean text",
        dataset_path="/tmp/a.jsonl",
        model_config_path="/tmp/models.yaml",
        options=None,
    )
    optimize_result = optimize_yaml_tool(
        yaml_text_or_path="process:\n  - clean_text_mapper: {}\n",
        objective="cost",
        model_config_path="/tmp/models.yaml",
        options=None,
    )

    assert generate_result.ok is True
    assert generate_result.error is None
    assert generate_result.data["options"] == {}
    assert generate_result.token_usage is None

    assert optimize_result.ok is True
    assert optimize_result.error is None
    assert optimize_result.data["options"] == {}
    assert optimize_result.token_usage is None


def test_router_generate_with_none_intent_reports_missing_required_argument() -> None:
    """Router should preserve required-field validation when intent is None."""
    router = Router()

    result = router.route(
        action="generate",
        payload={
            "intent": None,
            "dataset_path": "/tmp/a.jsonl",
            "model_config_path": "/tmp/models.yaml",
            "options": {},
        },
    )

    assert result.ok is False
    assert result.error == "missing required argument: intent"


def test_router_optimize_with_none_objective_reports_missing_required_argument() -> None:
    """Router should preserve required-field validation when objective is None."""
    router = Router()

    result = router.route(
        action="optimize",
        payload={
            "yaml_text_or_path": "process:\n  - clean_text_mapper: {}\n",
            "objective": None,
            "model_config_path": "/tmp/models.yaml",
            "options": {},
        },
    )

    assert result.ok is False
    assert result.error == "missing required argument: objective"


def test_router_required_text_fields_reject_non_string_values() -> None:
    """Router should return normalized type errors for non-string text fields."""
    router = Router()

    generate_result = router.route(
        action="generate",
        payload={
            "intent": 123,
            "dataset_path": "/tmp/a.jsonl",
            "model_config_path": "/tmp/models.yaml",
            "options": {},
        },
    )
    optimize_result = router.route(
        action="optimize",
        payload={
            "yaml_text_or_path": "process:\n  - clean_text_mapper: {}\n",
            "objective": ["quality"],
            "model_config_path": "/tmp/models.yaml",
            "options": {},
        },
    )
    explain_result = router.route(
        action="explain",
        payload={"operator_name": {"name": "clean_text_mapper"}, "options": {}},
    )

    assert generate_result.ok is False
    assert generate_result.error == "invalid type for intent"

    assert optimize_result.ok is False
    assert optimize_result.error == "invalid type for objective"

    assert explain_result.ok is False
    assert explain_result.error == "invalid type for operator_name"


def test_optimize_and_validate_tools_resolve_yaml_file_inputs_consistently(tmp_path: Path) -> None:
    """Optimize and validate tools should both resolve YAML path inputs."""
    yaml_path = tmp_path / "shared.yaml"
    yaml_path.write_text("process:\n  - clean_text_mapper: {}\n", encoding="utf-8")

    optimize_result = optimize_yaml_tool(
        yaml_text_or_path=str(yaml_path),
        objective="quality",
        model_config_path="/tmp/models.yaml",
        options={},
    )
    validate_result = validate_yaml_tool(yaml_text_or_path=str(yaml_path), options={})

    assert optimize_result.ok is True
    assert optimize_result.error is None
    assert optimize_result.data["optimized_yaml"] == "process:\n  - clean_text_mapper: {}\n"

    assert validate_result.ok is False
    assert validate_result.error == "validation failed"


def test_list_explain_and_validate_tools_handle_none_options() -> None:
    """Metadata/validation tools should normalize None options to empty mapping."""
    list_result = list_ops_tool(options=None)
    explain_result = explain_op_tool(operator_name="clean_text_mapper", options=None)
    validate_result = validate_yaml_tool(yaml_text_or_path="process: []", options=None)

    if list_result.ok:
        assert list_result.error is None
    else:
        assert list_result.error == "op search unavailable: data_juicer not installed"
    assert list_result.data["options"] == {}

    if explain_result.ok:
        assert explain_result.error is None
    else:
        assert explain_result.error in {
            "op search unavailable: data_juicer not installed",
            "operator not found: clean_text_mapper",
        }
    assert explain_result.data["options"] == {}

    assert validate_result.ok is False
    assert validate_result.error == "validation failed"
    assert validate_result.data["options"] == {}


def test_generate_and_optimize_tools_handle_truthy_non_mapping_options() -> None:
    """Generate/optimize tools should normalize truthy non-mapping options."""
    generate_result = generate_yaml_tool(
        intent="clean text",
        dataset_path="/tmp/a.jsonl",
        model_config_path="/tmp/models.yaml",
        options="bad-options",
    )
    optimize_result = optimize_yaml_tool(
        yaml_text_or_path="process:\n  - clean_text_mapper: {}\n",
        objective="cost",
        model_config_path="/tmp/models.yaml",
        options="bad-options",
    )

    assert generate_result.ok is True
    assert generate_result.error is None
    assert generate_result.data["options"] == {}

    assert optimize_result.ok is True
    assert optimize_result.error is None
    assert optimize_result.data["options"] == {}


def test_generate_tool_real_mode_uses_generator_runner(tmp_path: Path, monkeypatch) -> None:
    """Generate tool should use real runner when enabled."""

    def _fake_run_real_generation(**kwargs):
        return ({"process": [{"alpha_filter": {}}]}, {"total_tokens": 11})

    monkeypatch.setattr(
        "agent_runtime.tools.generate_yaml._run_real_generation",
        _fake_run_real_generation,
    )

    dataset_path = tmp_path / "input.jsonl"
    dataset_path.write_text('{"text":"x"}\n', encoding="utf-8")

    result = generate_yaml_tool(
        intent="do real generation",
        dataset_path=str(dataset_path),
        model_config_path="/tmp/models.yaml",
        options={"use_real_generator": True},
    )

    assert result.ok is True
    assert result.error is None
    assert "alpha_filter" in result.data["yaml_text"]
    assert result.token_usage == {"total_tokens": 11}


def test_generate_tool_real_mode_rejects_directory_dataset_path(tmp_path: Path) -> None:
    """Real generate mode should reject directory dataset paths."""
    result = generate_yaml_tool(
        intent="clean support tickets",
        dataset_path=str(tmp_path),
        model_config_path="/tmp/models.yaml",
        options={"use_real_generator": True},
    )

    assert result.ok is False
    assert result.error == "dataset_path must be a file path, not a directory"


def test_optimize_tool_real_mode_uses_optimizer_runner(monkeypatch) -> None:
    """Optimize tool should use real optimizer runner when enabled."""

    def _fake_run_real_optimization(**kwargs):
        return ("process:\n  - beta_mapper: {}\n", 3, ["minor warning"])

    monkeypatch.setattr(
        "agent_runtime.tools.optimize_yaml._run_real_optimization",
        _fake_run_real_optimization,
    )

    result = optimize_yaml_tool(
        yaml_text_or_path="process:\n  - alpha_filter: {}\n",
        objective="quality",
        model_config_path="/tmp/models.yaml",
        options={"use_real_optimizer": True},
    )

    assert result.ok is True
    assert result.error is None
    assert "beta_mapper" in result.data["optimized_yaml"]
    assert result.data["candidate_count"] == 3
    assert result.data["errors"] == ["minor warning"]


def test_list_explain_and_validate_tools_handle_truthy_non_mapping_options() -> None:
    """Metadata/validation tools should normalize truthy non-mapping options."""
    list_result = list_ops_tool(options="bad-options")
    explain_result = explain_op_tool(operator_name="clean_text_mapper", options="bad-options")
    validate_result = validate_yaml_tool(yaml_text_or_path="process: []", options="bad-options")

    if list_result.ok:
        assert list_result.error is None
    else:
        assert list_result.error == "op search unavailable: data_juicer not installed"
    assert list_result.data["options"] == {}

    if explain_result.ok:
        assert explain_result.error is None
    else:
        assert explain_result.error in {
            "op search unavailable: data_juicer not installed",
            "operator not found: clean_text_mapper",
        }
    assert explain_result.data["options"] == {}

    assert validate_result.ok is False
    assert validate_result.error == "validation failed"
    assert validate_result.data["options"] == {}
