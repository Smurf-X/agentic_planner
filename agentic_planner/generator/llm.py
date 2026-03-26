# -*- coding: utf-8 -*-
"""LLM protocol for JSON-only planner steps.

For production HTTP calls to OpenAI-compatible APIs, see
:class:`agentic_planner.generator.http_llm.OpenAICompatibleJsonClient`.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class LLMJsonClient(Protocol):
    """Minimal adapter: given a prompt, return a parsed JSON object."""

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        """Return a JSON object (dict)."""


class DictLLMJsonClient:
    """Test double: returns a fixed dict (ignores prompts)."""

    def __init__(self, payload: Dict[str, Any] | None = None) -> None:
        self.payload = payload or {}

    def complete_json(self, system: str, user: str) -> Dict[str, Any]:
        return dict(self.payload)


def parse_json_object_strict(text: str) -> Dict[str, Any]:
    """Parse JSON object from model output; strip fences if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) >= 2:
            inner = parts[1]
            if inner.startswith("json"):
                inner = inner[4:]
            stripped = inner.strip()
    data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data