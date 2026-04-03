# -*- coding: utf-8 -*-
"""Boundary tests for TUI import restrictions."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List


def _collect_tui_python_files() -> List[Path]:
    root = Path(__file__).resolve().parents[1]
    return sorted((root / "apps" / "tui").rglob("*.py"))


def _imported_modules(path: Path) -> List[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                modules.append(name.name)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    return modules


def test_tui_does_not_import_agentic_planner_directly() -> None:
    """TUI modules must not access agentic_planner directly."""
    for path in _collect_tui_python_files():
        modules = _imported_modules(path)
        banned = [module for module in modules if module.startswith("agentic_planner")]
        assert banned == [], f"{path} has banned imports: {banned}"


def test_tui_runtime_access_uses_service_boundary() -> None:
    """Any agent_runtime import in TUI must go through service boundary."""
    allowed_runtime_import = "agent_runtime.api.service"
    for path in _collect_tui_python_files():
        modules = _imported_modules(path)
        runtime_imports = [module for module in modules if module.startswith("agent_runtime")]
        disallowed = [module for module in runtime_imports if module != allowed_runtime_import]
        assert disallowed == [], f"{path} has disallowed runtime imports: {disallowed}"
