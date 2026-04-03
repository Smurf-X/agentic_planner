# -*- coding: utf-8 -*-
"""Runnable TUI shell for agent runtime MVP."""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from textual.app import App as TextualBaseApp
    from textual.app import ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Footer, Header, Input, RichLog, Static

    TEXTUAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    TEXTUAL_AVAILABLE = False

    class TextualBaseApp:  # type: ignore[no-redef]
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


@dataclass
class SessionContext:
    """Mutable command context for both plain and textual UIs."""

    objective: str
    dataset_path: str
    model_config_path: str
    model: str
    base_url: str
    api_key: str
    max_iterations: int
    current_yaml: str = ""


class AgentPlannerTUI(TextualBaseApp):
    """TUI shell with a Textual UI when available and plain fallback otherwise."""

    if TEXTUAL_AVAILABLE:
        CSS = """
        Screen {
            background: #12141a;
            color: #eaf0ff;
        }
        Header {
            background: #1f2432;
            color: #eaf0ff;
            text-style: bold;
        }
        Footer {
            background: #1a1f2b;
            color: #aeb8d4;
        }
        #layout {
            height: 1fr;
            padding: 1 2;
            background: #12141a;
        }
        #left-pane {
            width: 38;
            border: round #2d3448;
            padding: 1 2;
            background: #1a1f2b;
            margin-right: 1;
        }
        #right-pane {
            width: 1fr;
            border: round #2d3448;
            padding: 0;
            background: #202738;
        }
        #output-title {
            height: 1;
            color: #ff7a59;
            background: #1a1f2b;
            text-style: bold;
            padding: 0 1;
            border-bottom: solid #2d3448;
        }
        #output {
            background: #202738;
            color: #eaf0ff;
            padding: 1;
        }
        #menu-title {
            color: #ff7a59;
            text-style: bold;
            margin-bottom: 1;
        }
        #menu-body {
            color: #c2cbea;
        }
        #status {
            height: auto;
            color: #12141a;
            background: #38bdf8;
            text-style: bold;
            padding: 0 2;
            margin: 0 2;
        }
        #command-input {
            dock: bottom;
            background: #161b27;
            color: #f5f8ff;
            border-top: solid #2d3448;
            padding: 0 1;
        }
        #command-input:focus {
            border-top: solid #ff7a59;
            background: #1c2232;
        }
        """

    def __init__(self, service: Optional[RuntimeServiceLike] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.service: RuntimeServiceLike = service or create_runtime_service()
        self._routes: Dict[str, AppRoute] = {
            "generate": AppRoute(name="generate", screen=WorkflowScreen),
            "optimize": AppRoute(name="optimize", screen=WorkflowScreen),
            "operator": AppRoute(name="operator", screen=OperatorScreen),
            "chat": AppRoute(name="chat", screen=ChatScreen),
            "export": AppRoute(name="export", screen=ExportScreen),
        }
        self.workflow = self.open_route("generate")
        self.operator = self.open_route("operator")
        self.chat = self.open_route("chat")

        self.ctx = SessionContext(
            objective="quality",
            dataset_path="",
            model_config_path="",
            model="",
            base_url="",
            api_key="",
            max_iterations=3,
        )

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

    if TEXTUAL_AVAILABLE:

        def compose(self) -> ComposeResult:
            """Compose Textual layout."""
            yield Header(show_clock=True)
            yield Static("", id="status")
            with Horizontal(id="layout"):
                with Vertical(id="left-pane"):
                    yield Static("Agentic Planner", id="menu-title")
                    yield Static(
                        "\n".join(
                            [
                                "Commands",
                                "  help / h / ?",
                                "  generate <intent>",
                                "  optimize",
                                "  validate",
                                "  ops",
                                "  op <name>",
                                "",
                                "Set context",
                                "  set dataset <path>",
                                "  set model_config <path>",
                                "  set model <name>",
                                "  set objective <quality|cost|balanced>",
                                "",
                                "Exit",
                                "  quit / q",
                            ]
                        ),
                        id="menu-body",
                    )
                with Vertical(id="right-pane"):
                    yield Static("Session Output", id="output-title")
                    yield RichLog(highlight=True, markup=False, wrap=True, id="output")
            yield Input(placeholder="Type command, e.g. generate 清洗客服工单", id="command-input")
            yield Footer()

        def on_mount(self) -> None:
            """Initialize view on mount."""
            self._refresh_status()
            self._write_lines([
                "Agentic Planner Textual UI ready.",
                "Type `help` to view commands.",
            ])

        def on_input_submitted(self, event: Input.Submitted) -> None:
            """Handle command input in Textual mode."""
            raw = event.value.strip()
            event.input.value = ""
            if not raw:
                return

            lines, should_quit = execute_command(
                raw,
                ctx=self.ctx,
                workflow=self.workflow,
                operator=self.operator,
                chat=self.chat,
            )
            self._write_lines([f"> {raw}"] + lines)
            self._refresh_status()
            if should_quit:
                self.exit()

        def _refresh_status(self) -> None:
            """Refresh status strip text."""
            status = build_status_line(
                objective=self.ctx.objective,
                dataset_path=self.ctx.dataset_path,
                model_config_path=self.ctx.model_config_path,
                model=self.ctx.model,
            )
            self.query_one("#status", Static).update(status)

        def _write_lines(self, lines: List[str]) -> None:
            """Append lines to output log."""
            output = self.query_one("#output", RichLog)
            for line in lines:
                output.write(line)


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
    parser.add_argument("--plain", action="store_true", help="Use plain prompt-loop instead of Textual UI")
    return parser


def build_help_text() -> str:
    """Build compact help text shown in the command loop."""
    return (
        "Commands:\n"
        "  generate <intent>      - Build YAML from intent\n"
        "  optimize               - Optimize current YAML\n"
        "  ops                    - List operators\n"
        "  op <name>              - Explain one operator\n"
        "  validate               - Validate current YAML\n"
        "  set dataset <path>     - Set default dataset path\n"
        "  set model_config <p>   - Set default models.yaml path\n"
        "  set model <name>       - Set preferred model\n"
        "  set objective <value>  - quality | cost | balanced\n"
        "  help (aliases: h, ?)   - Show this help\n"
        "  quit (alias: q)        - Exit\n"
        "\n"
        "Examples:\n"
        "  set dataset /data/support_tickets.jsonl\n"
        "  set model_config /repo/models.yaml\n"
        "  generate 清洗客服工单并去重\n"
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


def configure_runtime_environment() -> None:
    """Reduce noisy logs/warnings for interactive terminal usage."""
    warnings.filterwarnings(
        "ignore",
        message=r"invalid escape sequence.*",
        category=SyntaxWarning,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("data_juicer").setLevel(logging.WARNING)


def _safe_input(prompt: str) -> Optional[str]:
    """Read input and return None on EOF for graceful exit."""
    try:
        return input(prompt)
    except EOFError:
        return None


def _prompt_with_default(label: str, default: str) -> Optional[str]:
    """Prompt user with a default value and return selected text or None on EOF."""
    suffix = f" [{default}]" if default else ""
    value = _safe_input(f"{label}{suffix}: ")
    if value is None:
        return None
    value = value.strip()
    if value:
        return value
    return default


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


def _build_form_from_context(ctx: SessionContext, intent: str) -> WorkflowFormData:
    """Construct workflow form from current command context."""
    return WorkflowFormData(
        task_description=intent,
        dataset_path=ctx.dataset_path,
        optimization_preference=ctx.objective,
        model_config_path=ctx.model_config_path,
        llm_model=ctx.model,
        llm_base_url=ctx.base_url,
        llm_api_key=ctx.api_key,
        max_iterations=ctx.max_iterations,
    )


def execute_command(
    raw: str,
    *,
    ctx: SessionContext,
    workflow: WorkflowScreen,
    operator: OperatorScreen,
    chat: ChatScreen,
) -> Tuple[List[str], bool]:
    """Execute one command and return output lines plus quit flag."""
    command = normalize_command(raw)
    if not command:
        return ([], False)

    if command == "help":
        return (build_help_text().splitlines(), False)
    if command == "quit":
        return (["Goodbye."], True)

    if command.startswith("set "):
        parts = command.split(maxsplit=2)
        if len(parts) < 3:
            return (["[ERROR] Usage: set <dataset|model_config|model|objective> <value>"], False)
        key = parts[1].strip()
        value = parts[2].strip()
        if key == "dataset":
            validation_error = validate_dataset_path(value)
            if validation_error is not None:
                return ([f"[ERROR] {validation_error}"], False)
            ctx.dataset_path = value
            return ([f"[OK] dataset path set: {value}"], False)
        if key == "model_config":
            ctx.model_config_path = value
            return ([f"[OK] model config set: {value}"], False)
        if key == "model":
            ctx.model = value
            return ([f"[OK] model set: {value}"], False)
        if key == "objective":
            if value not in {"quality", "cost", "balanced"}:
                return (["[ERROR] objective must be one of: quality, cost, balanced"], False)
            ctx.objective = value
            return ([f"[OK] objective set: {value}"], False)
        return ([f"[ERROR] unsupported set target: {key}"], False)

    if command == "ops":
        return (render_response_lines(operator.list_operators()), False)

    if command.startswith("op "):
        name = command[3:].strip()
        if not name:
            return (["[ERROR] Usage: op <operator_name>"], False)
        return (render_response_lines(operator.explain_operator(name)), False)

    if command == "validate":
        if not ctx.current_yaml:
            return (["[ERROR] No YAML in session. Run generate first or set current YAML."], False)
        return (render_response_lines(workflow.validate_yaml(ctx.current_yaml)), False)

    if command == "optimize":
        if not ctx.current_yaml:
            return (["[ERROR] No YAML in session. Run generate first."], False)
        form = _build_form_from_context(ctx, intent="")
        result = workflow.submit_optimize(form, yaml_text=ctx.current_yaml)
        if result.ok:
            ctx.current_yaml = str(result.data.get("optimized_yaml", ctx.current_yaml))
        return (render_response_lines(result), False)

    if command.startswith("generate"):
        intent = command[len("generate") :].strip()
        if not intent:
            return (["[ERROR] Usage: generate <natural-language-intent>"], False)
        if not ctx.dataset_path:
            return (["[ERROR] dataset path is unset. Use: set dataset <path>"], False)
        form = _build_form_from_context(ctx, intent=intent)
        result = workflow.submit_generate(form)
        if result.ok:
            ctx.current_yaml = str(result.data.get("yaml_text", ""))
        return (render_response_lines(result), False)

    if command.startswith("/"):
        return (render_response_lines(chat.submit_message(command)), False)

    return (["[ERROR] Unknown command. Type `help` for usage."], False)


def run_interactive_cli(args: argparse.Namespace) -> int:
    """Run plain prompt-loop UI (fallback and optional mode)."""
    configure_runtime_environment()
    app = AgentPlannerTUI()
    workflow = app.open_route("generate")
    operator = app.open_route("operator")
    chat = app.open_route("chat")

    assert isinstance(workflow, WorkflowScreen)
    assert isinstance(operator, OperatorScreen)
    assert isinstance(chat, ChatScreen)

    app.ctx.objective = args.objective
    app.ctx.dataset_path = args.dataset_path
    app.ctx.model_config_path = args.model_config_path
    app.ctx.model = args.model
    app.ctx.base_url = args.base_url
    app.ctx.api_key = args.api_key
    app.ctx.max_iterations = args.max_iterations

    print("Agentic Planner TUI ready")
    print("Type 'help' to see commands.")
    print("Tip: this plain mode supports one-line commands. Example: generate 清洗客服工单并去重")

    while True:
        print(
            build_status_line(
                objective=app.ctx.objective,
                dataset_path=app.ctx.dataset_path,
                model_config_path=app.ctx.model_config_path,
                model=app.ctx.model,
            )
        )
        raw_input = _safe_input("tui> ")
        if raw_input is None:
            print("\nExiting TUI (EOF).")
            return 0
        lines, should_quit = execute_command(
            raw_input.strip(),
            ctx=app.ctx,
            workflow=workflow,
            operator=operator,
            chat=chat,
        )
        for line in lines:
            print(line)
        if should_quit:
            return 0


def main() -> int:
    """CLI entrypoint."""
    configure_runtime_environment()
    parser = _build_parser()
    args = parser.parse_args()

    if TEXTUAL_AVAILABLE and not args.plain:
        app = AgentPlannerTUI()
        app.ctx.objective = args.objective
        app.ctx.dataset_path = args.dataset_path
        app.ctx.model_config_path = args.model_config_path
        app.ctx.model = args.model
        app.ctx.base_url = args.base_url
        app.ctx.api_key = args.api_key
        app.ctx.max_iterations = args.max_iterations
        app.run()
        return 0

    return run_interactive_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
