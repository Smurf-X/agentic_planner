# -*- coding: utf-8 -*-
"""Smoke tests for frontend skeleton."""
import subprocess
import sys
from pathlib import Path


def test_frontend_package_json_exists():
    pkg_path = Path("apps/webui/frontend/package.json")
    assert pkg_path.exists()


def test_frontend_vite_config_exists():
    config_path = Path("apps/webui/frontend/vite.config.ts")
    assert config_path.exists()


def test_frontend_src_main_exists():
    main_path = Path("apps/webui/frontend/src/main.tsx")
    assert main_path.exists()