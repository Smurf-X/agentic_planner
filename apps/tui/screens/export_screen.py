# -*- coding: utf-8 -*-
"""Export UX placeholders for YAML preview/copy flows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OverwritePreview:
    """Placeholder overwrite preview state."""

    changed: bool
    message: str


class ExportScreen:
    """Lightweight export helpers used by the TUI shell."""

    def preview_before_overwrite(self, existing_yaml: str, next_yaml: str) -> OverwritePreview:
        """Return deterministic overwrite preview text for UI wiring."""
        changed = existing_yaml != next_yaml
        if changed:
            message = "Preview before overwrite: YAML content will be replaced."
        else:
            message = "Preview before overwrite: no changes detected."
        return OverwritePreview(changed=changed, message=message)

    def copy_yaml_once(self, yaml_text: str) -> str:
        """One-action copy placeholder that returns payload for clipboard adapter."""
        return yaml_text
