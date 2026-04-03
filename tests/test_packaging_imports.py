# -*- coding: utf-8 -*-
"""Packaging-level import smoke tests for runtime and TUI entrypoint."""


def test_installed_packages_include_runtime_and_tui_imports() -> None:
    """Ensure runtime and TUI packages are importable."""
    import agent_runtime  # noqa: F401
    from apps.tui.main import AgentPlannerTUI  # noqa: F401
