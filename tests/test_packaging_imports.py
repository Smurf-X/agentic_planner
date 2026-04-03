# -*- coding: utf-8 -*-
"""Packaging-level wheel content tests for runtime and TUI files."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path


def test_built_wheel_includes_runtime_and_tui_files(tmp_path: Path) -> None:
    """Build a wheel and verify expected runtime and TUI files are packaged."""
    repo_root = Path(__file__).resolve().parents[1]
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--wheel-dir",
            str(dist_dir),
        ],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    wheel_paths = sorted(dist_dir.glob("*.whl"))
    assert wheel_paths, "Expected pip wheel to produce a wheel file"

    with zipfile.ZipFile(wheel_paths[0]) as wheel_zip:
        packaged_files = set(wheel_zip.namelist())

    assert "agent_runtime/__init__.py" in packaged_files
    assert "apps/tui/main.py" in packaged_files
