# -*- coding: utf-8 -*-
"""Smoke tests for TUI bootstrap and top-level routes."""

import logging

import pytest

from apps.tui.main import (
    _safe_input,
    AgentPlannerTUI,
    build_status_line,
    build_help_text,
    configure_runtime_environment,
    normalize_command,
    render_response_lines,
    validate_dataset_path,
)
from apps.tui.screens.chat_screen import ChatScreen
from apps.tui.screens.export_screen import ExportScreen
from apps.tui.screens.operator_screen import OperatorScreen
from apps.tui.screens.workflow_screen import WorkflowScreen


def test_tui_app_imports() -> None:
    """App class should be importable for CLI bootstrap."""
    assert AgentPlannerTUI is not None


def test_tui_exposes_required_menu_routes() -> None:
    """Top-level TUI should expose all MVP routes."""
    app = AgentPlannerTUI()

    assert app.get_menu_routes() == ["generate", "optimize", "operator", "chat", "export"]


def test_open_route_returns_expected_screen_types() -> None:
    """Route opening should instantiate expected screen classes."""
    app = AgentPlannerTUI()

    assert isinstance(app.open_route("generate"), WorkflowScreen)
    assert isinstance(app.open_route("optimize"), WorkflowScreen)
    assert isinstance(app.open_route("operator"), OperatorScreen)
    assert isinstance(app.open_route("chat"), ChatScreen)
    assert isinstance(app.open_route("export"), ExportScreen)


def test_open_route_raises_for_unknown_route() -> None:
    """Unknown route names should raise a ValueError."""
    app = AgentPlannerTUI()

    with pytest.raises(ValueError, match="unknown route"):
        app.open_route("unknown")


def test_configure_runtime_environment_sets_quiet_logger_levels() -> None:
    """Runtime environment setup should suppress noisy HTTPX logs."""
    configure_runtime_environment()

    assert logging.getLogger("httpx").level == logging.WARNING


def test_help_text_includes_examples_and_commands() -> None:
    """Help output should include command list and examples."""
    help_text = build_help_text()

    assert "generate" in help_text
    assert "optimize" in help_text
    assert "Examples" in help_text


def test_normalize_command_supports_aliases() -> None:
    """Command aliases should normalize to canonical commands."""
    assert normalize_command("h") == "help"
    assert normalize_command("?") == "help"
    assert normalize_command("g") == "generate"
    assert normalize_command("o") == "optimize"


def test_validate_dataset_path_rejects_missing_and_directory(tmp_path) -> None:
    """Dataset validator should reject missing paths and directories."""
    missing = validate_dataset_path(str(tmp_path / "missing.jsonl"))
    directory = validate_dataset_path(str(tmp_path))

    assert missing == "dataset_path does not exist"
    assert directory == "dataset_path must be a file path, not a directory"


def test_build_status_line_contains_context() -> None:
    """Status line should include objective and configured model context."""
    status = build_status_line(
        objective="quality",
        dataset_path="/tmp/data.jsonl",
        model_config_path="/tmp/models.yaml",
        model="kimi-k2.5",
    )

    assert "objective=quality" in status
    assert "model=kimi-k2.5" in status


def test_safe_input_returns_none_on_eof(monkeypatch) -> None:
    """Safe input helper should convert EOFError into None."""

    def _raise_eof(prompt: str) -> str:  # noqa: ARG001
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)
    assert _safe_input("tui> ") is None


def test_render_response_lines_formats_yaml_preview_and_suggestion() -> None:
    """Renderer should show YAML preview and actionable error suggestion."""

    class _Resp:
        ok = False
        error = "dataset_path must be a file path, not a directory"
        data = {"yaml_text": "a\nb\nc\nd\n", "other": 1}

    lines = render_response_lines(_Resp(), yaml_preview_lines=2)

    assert lines[0] == "[ERROR] dataset_path must be a file path, not a directory"
    assert any(line.startswith("Suggestion:") for line in lines)
    assert any("yaml_text" in line for line in lines)
    assert any("..." in line for line in lines)


def test_render_response_lines_formats_operator_list_panel() -> None:
    """Renderer should show compact operator panel output."""

    class _Resp:
        ok = True
        error = None
        data = {
            "operators": [
                {
                    "name": "alpha_filter",
                    "category": "filter",
                    "summary": "Filter low-quality rows.",
                },
                {
                    "name": "beta_mapper",
                    "category": "mapper",
                    "summary": "Normalize text fields.",
                },
            ]
        }

    lines = render_response_lines(_Resp())

    assert any(line.startswith("Operators (2):") for line in lines)
    assert any("alpha_filter [filter]" in line for line in lines)
    assert any("beta_mapper [mapper]" in line for line in lines)


def test_render_response_lines_formats_validate_panel() -> None:
    """Renderer should show validation status and error list."""

    class _Resp:
        ok = False
        error = "validation failed"
        data = {
            "valid": False,
            "errors": ["unknown operator 'x'", "params must be dict"],
            "options": {"route": "validate"},
        }

    lines = render_response_lines(_Resp())

    assert any(line == "Validation: INVALID" for line in lines)
    assert any(line.startswith("- unknown operator") for line in lines)
    assert any(line.startswith("- params must be dict") for line in lines)


def test_render_response_lines_formats_operator_detail_panel() -> None:
    """Renderer should format single-operator explanation in fixed order."""

    class _Resp:
        ok = True
        error = None
        data = {
            "name": "alpha_filter",
            "category": "filter",
            "tags": ["quality"],
            "signature": "(threshold: float = 0.8)",
            "summary": "Filter low-quality rows.",
            "param_desc": "threshold: keep rows with score >= threshold",
            "options": {"route": "operator"},
        }

    lines = render_response_lines(_Resp())

    assert any(line == "Operator: alpha_filter" for line in lines)
    assert any(line == "Category: filter" for line in lines)
    assert any(line.startswith("Signature:") for line in lines)
    assert any(line.startswith("Summary:") for line in lines)
    assert any(line.startswith("Parameters:") for line in lines)
