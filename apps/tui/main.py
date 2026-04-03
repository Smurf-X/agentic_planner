# -*- coding: utf-8 -*-
"""Runnable TUI shell for agent runtime MVP."""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from textual.app import App as TextualApp
except ImportError:  # pragma: no cover
    class TextualApp:  # type: ignore[no-redef]
        """Fallback base app for environments without textual installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


from apps.tui.runtime_boundary import RuntimeServiceLike, create_runtime_service
from apps.tui.screens.chat_screen import ChatScreen
from apps.tui.screens.export_screen import ExportScreen
from apps.tui.screens.operator_screen import OperatorScreen
from apps.tui.screens.workflow_screen import WorkflowFormData, WorkflowScreen


@dataclass
class AppRoute:
    """Declarative route mapping for the TUI shell."""

    name: str
    screen: Any


class AgentPlannerTUI(TextualApp):
    """Minimal TUI shell that exposes MVP menu routes."""

    def __init__(self, service: Optional[RuntimeServiceLike] = None) -> None:
        super().__init__()
        self.service: RuntimeServiceLike = service or create_runtime_service()
        self._routes: Dict[str, AppRoute] = {
            "generate": AppRoute(name="generate", screen=WorkflowScreen),
            "optimize": AppRoute(name="optimize", screen=WorkflowScreen),
            "operator": AppRoute(name="operator", screen=OperatorScreen),
            "chat": AppRoute(name="chat", screen=ChatScreen),
            "export": AppRoute(name="export", screen=ExportScreen),
        }

    def get_menu_routes(self) -> List[str]:
        """Return top-level MVP routes in menu order."""
        return ["generate", "optimize", "operator", "chat", "export"]

    def open_route(self, route_name: str) -> Any:
        """Instantiate and return a screen by route name."""
        route = self._routes.get(route_name)
        if route is None:
            raise ValueError(f"unknown route: {route_name}")

        if route.screen in {WorkflowScreen, OperatorScreen, ChatScreen}:
            return route.screen(service=self.service)
        return route.screen()


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for interactive TUI runner."""
    parser = argparse.ArgumentParser(description="Agentic Planner TUI (interactive MVP)")
    parser.add_argument("--dataset-path", default="", help="Input dataset path")
    parser.add_argument("--model-config-path", default="", help="models.yaml path for model registry")
    parser.add_argument("--model", default="", help="Direct model name override")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--objective", default="quality", help="Optimize objective")
    parser.add_argument("--max-iterations", type=int, default=3, help="Optimizer max iterations")
    return parser


def build_help_text() -> str:
    """Build compact help text shown in the command loop."""
    return (
        "Commands:\n"
        "  generate (alias: g)  - Build YAML from intent\n"
        "  optimize (alias: o)  - Optimize current YAML\n"
        "  ops                  - List operators\n"
        "  op <name>            - Explain one operator\n"
        "  validate (alias: v)  - Validate current YAML\n"
        "  help (aliases: h, ?) - Show this help\n"
        "  quit (alias: q)      - Exit\n"
        "\n"
        "Examples:\n"
        "  generate\n"
        "  op language_id_score_filter\n"
        "  optimize\n"
    )


def normalize_command(raw: str) -> str:
    """Normalize command aliases to canonical route names."""
    text = raw.strip()
    aliases = {
        "h": "help",
        "?": "help",
        "g": "generate",
        "o": "optimize",
        "v": "validate",
        "q": "quit",
        "exit": "quit",
    }
    return aliases.get(text, text)


def validate_dataset_path(dataset_path: str) -> Optional[str]:
    """Validate dataset path for interactive form input."""
    text = dataset_path.strip()
    if not text:
        return "dataset_path is required"

    candidate = Path(text)
    if not candidate.exists():
        return "dataset_path does not exist"
    if candidate.is_dir():
        return "dataset_path must be a file path, not a directory"
    return None


def build_status_line(
    *,
    objective: str,
    dataset_path: str,
    model_config_path: str,
    model: str,
) -> str:
    """Render a compact status line for the interactive shell."""
    dataset_hint = dataset_path or "<unset>"
    model_cfg_hint = model_config_path or "<unset>"
    model_hint = model or "<auto>"
    return (
        "status: "
        f"objective={objective} | "
        f"dataset={dataset_hint} | "
        f"model_config={model_cfg_hint} | "
        f"model={model_hint}"
    )


def _prompt_with_default(label: str, default: str) -> str:
    """Prompt user with a default value and return selected text."""
    suffix = f" [{default}]" if default else ""
    value = _safe_input(f"{label}{suffix}: ")
    if value is None:
        raise EOFError
    value = value.strip()
    if value:
        return value
    return default


def _safe_input(prompt: str) -> Optional[str]:
    """Read input and return None on EOF for graceful exit."""
    try:
        return input(prompt)
    except EOFError:
        return None


def configure_runtime_environment() -> None:
    """Reduce noisy logs/warnings for interactive terminal usage."""
    warnings.filterwarnings(
        "ignore",
        message=r"invalid escape sequence.*",
        category=SyntaxWarning,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("data_juicer").setLevel(logging.WARNING)


def _print_response(result: Any) -> None:
    """Print normalized tool response for interactive CLI."""
    for line in render_response_lines(result):
        print(line)


def render_response_lines(result: Any, yaml_preview_lines: int = 40) -> List[str]:
    """Render structured response lines for terminal display."""
    status = "OK" if result.ok else "ERROR"
    error_text = str(result.error or "").strip()
    lines: List[str] = [f"[{status}] {error_text}".rstrip()]

    if not getattr(result, "ok", False) and error_text:
        suggestion = _suggestion_for_error(error_text)
        if suggestion:
            lines.append(f"Suggestion: {suggestion}")

    data = getattr(result, "data", {})
    if not isinstance(data, dict) or not data:
        return lines

    panel_lines = _render_specialized_panel(data)
    if panel_lines:
        lines.extend(panel_lines)
        return lines

    for key, value in data.items():
        if key in {"yaml_text", "optimized_yaml"}:
            lines.append(f"{key}:")
            yaml_lines = str(value).splitlines()
            preview = yaml_lines[:yaml_preview_lines]
            lines.extend(preview)
            hidden_count = len(yaml_lines) - len(preview)
            if hidden_count > 0:
                lines.append(f"... ({hidden_count} more lines)")
            continue
        lines.append(f"{key}: {value}")
    return lines


def _render_specialized_panel(data: Dict[str, Any]) -> List[str]:
    """Render specialized views for operator and validation payloads."""
    if "operators" in data and isinstance(data.get("operators"), list):
        operators = data.get("operators") or []
        lines: List[str] = [f"Operators ({len(operators)}):"]
        for item in operators:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "<unknown>"))
            category = str(item.get("category", ""))
            summary = str(item.get("summary", "")).strip()
            header = f"- {name}"
            if category:
                header = f"{header} [{category}]"
            lines.append(header)
            if summary:
                lines.append(f"  {summary}")
        return lines

    if "valid" in data and "errors" in data:
        valid = bool(data.get("valid", False))
        status = "VALID" if valid else "INVALID"
        lines = [f"Validation: {status}"]
        errors = data.get("errors")
        if isinstance(errors, list) and errors:
            lines.append("Errors:")
            for err in errors:
                lines.append(f"- {err}")
        return lines

    if "name" in data and ("signature" in data or "param_desc" in data):
        name = str(data.get("name", "<unknown>"))
        category = str(data.get("category", "")).strip()
        signature = str(data.get("signature", "")).strip()
        summary = str(data.get("summary", "")).strip()
        param_desc = str(data.get("param_desc", "")).strip()
        tags = data.get("tags")

        lines = [f"Operator: {name}"]
        if category:
            lines.append(f"Category: {category}")
        if isinstance(tags, list) and tags:
            lines.append(f"Tags: {', '.join(str(tag) for tag in tags)}")
        if signature:
            lines.append(f"Signature: {signature}")
        if summary:
            lines.append(f"Summary: {summary}")
        if param_desc:
            lines.append(f"Parameters: {param_desc}")
        return lines

    return []


def _suggestion_for_error(error: str) -> str:
    """Map common runtime errors to actionable next steps."""
    lower = error.lower()
    if "dataset_path must be a file path" in lower:
        return "Provide a .jsonl file path like /path/to/input.jsonl."
    if "dataset_path does not exist" in lower:
        return "Check the path and file name, then retry generate."
    if "api_key is required" in lower:
        return "Pass --api-key, set OPENAI_API_KEY, or use --model-config-path."
    if "model config file not found" in lower:
        return "Check --model-config-path and ensure the YAML file exists."
    if "operator not found" in lower:
        return "Use `ops` first, then `op <name>` with an exact operator name."
    if "validation failed" in lower:
        return "Run `validate` and inspect unknown operators/params in errors."
    return "Review input values and retry; run `help` for command examples."


def run_interactive_cli(args: argparse.Namespace) -> int:
    """Run a simple command-loop TUI for MVP usage."""
    configure_runtime_environment()
    app = AgentPlannerTUI()
    workflow = app.open_route("generate")
    assert isinstance(workflow, WorkflowScreen)
    operator = app.open_route("operator")
    assert isinstance(operator, OperatorScreen)

    current_yaml = ""
    dataset_path_value = args.dataset_path
    model_config_value = args.model_config_path
    print("Agentic Planner TUI ready")
    print("Type 'help' to see commands.")
    print("Tip: use a JSONL file path for dataset_path, not a directory")
    while True:
        print(
            build_status_line(
                objective=args.objective,
                dataset_path=dataset_path_value,
                model_config_path=model_config_value,
                model=args.model,
            )
        )
        raw_input = _safe_input("tui> ")
        if raw_input is None:
            print("\nExiting TUI (EOF).")
            return 0
        raw = raw_input.strip()
        command = normalize_command(raw)
        if not command:
            continue
        if command == "help":
            print(build_help_text())
            continue
        if command == "quit":
            return 0
        if command == "ops":
            _print_response(operator.list_operators())
            continue
        if command.startswith("op "):
            _print_response(operator.explain_operator(raw[3:].strip()))
            continue
        if command == "generate":
            intent_input = _safe_input("intent> Describe your pipeline goal (natural language): ")
            if intent_input is None:
                print("\nGenerate cancelled (EOF).")
                return 0
            intent = intent_input.strip()
            try:
                while True:
                    dataset_path = _prompt_with_default(
                        "dataset_path> Input .jsonl file path",
                        dataset_path_value,
                    )
                    validation_error = validate_dataset_path(dataset_path)
                    if validation_error is None:
                        dataset_path_value = dataset_path
                        break
                    print(f"[ERROR] {validation_error}")

                model_config_path = _prompt_with_default(
                    "model_config_path> models.yaml path (optional)",
                    model_config_value,
                )
            except EOFError:
                print("\nGenerate cancelled (EOF).")
                return 0
            model_config_value = model_config_path
            form = WorkflowFormData(
                task_description=intent,
                dataset_path=dataset_path,
                optimization_preference=args.objective,
                model_config_path=model_config_path,
                llm_model=args.model,
                llm_base_url=args.base_url,
                llm_api_key=args.api_key,
                max_iterations=args.max_iterations,
            )
            result = workflow.submit_generate(form)
            _print_response(result)
            if result.ok:
                current_yaml = str(result.data.get("yaml_text", ""))
            continue
        if command == "optimize":
            if not current_yaml:
                yaml_input = _safe_input("yaml_text_or_path> Provide YAML text or file path: ")
                if yaml_input is None:
                    print("\nOptimize cancelled (EOF).")
                    return 0
                current_yaml = yaml_input.strip()
            form = WorkflowFormData(
                task_description="",
                dataset_path=dataset_path_value,
                optimization_preference=args.objective,
                model_config_path=model_config_value,
                llm_model=args.model,
                llm_base_url=args.base_url,
                llm_api_key=args.api_key,
                max_iterations=args.max_iterations,
            )
            result = workflow.submit_optimize(form, yaml_text=current_yaml)
            _print_response(result)
            if result.ok:
                current_yaml = str(result.data.get("optimized_yaml", current_yaml))
            continue
        if command == "validate":
            if not current_yaml:
                yaml_input = _safe_input("yaml_text_or_path> Provide YAML text or file path: ")
                if yaml_input is None:
                    print("\nValidate cancelled (EOF).")
                    return 0
                current_yaml = yaml_input.strip()
            _print_response(workflow.validate_yaml(current_yaml))
            continue
        print("Unknown command. Use: generate, optimize, ops, op <name>, validate, quit")


def main() -> int:
    """CLI entrypoint."""
    configure_runtime_environment()
    parser = _build_parser()
    args = parser.parse_args()
    return run_interactive_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
