# -*- coding: utf-8 -*-
"""Result panel state container for deterministic UI tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ResultPanel:
    """Store latest result and a lightweight history for screen rendering."""

    latest: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def push(self, result: Dict[str, Any]) -> None:
        """Append result to history and update latest payload."""
        self.latest = dict(result)
        self.history.append(dict(result))
