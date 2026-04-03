# -*- coding: utf-8 -*-
"""Export UX placeholders for YAML preview/copy flows."""

from __future__ import annotations

from dataclasses import dataclass


PREVIEW_CHANGED_MESSAGE = "Preview before overwrite: YAML content will be replaced."
PREVIEW_UNCHANGED_MESSAGE = "Preview before overwrite: no changes detected."


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
        message = PREVIEW_CHANGED_MESSAGE if changed else PREVIEW_UNCHANGED_MESSAGE
        return OverwritePreview(changed=changed, message=message)

    def copy_yaml_once(self, yaml_text: str) -> str:
        """One-action copy placeholder that returns payload for clipboard adapter."""
        return yaml_text
