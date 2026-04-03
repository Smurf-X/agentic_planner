# -*- coding: utf-8 -*-
"""Session state for agent runtime orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionState:
    """In-memory session state for runtime interactions."""

    session_id: str
    current_yaml: str = ""
    last_generated_yaml: str = ""
    last_optimized_candidates: List[Dict[str, Any]] = field(default_factory=list)
    objective: Optional[str] = None
    dataset_path: Optional[str] = None
    model_config_path: Optional[str] = None
    event_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session state into a JSON-friendly dictionary."""
        return {
            "session_id": self.session_id,
            "current_yaml": self.current_yaml,
            "last_generated_yaml": self.last_generated_yaml,
            "last_optimized_candidates": self.last_optimized_candidates,
            "objective": self.objective,
            "dataset_path": self.dataset_path,
            "model_config_path": self.model_config_path,
            "event_history": self.event_history,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SessionState":
        """Build session state from persisted payload."""
        return cls(
            session_id=str(payload.get("session_id", "")),
            current_yaml=str(payload.get("current_yaml", "")),
            last_generated_yaml=str(payload.get("last_generated_yaml", "")),
            last_optimized_candidates=list(payload.get("last_optimized_candidates", [])),
            objective=payload.get("objective"),
            dataset_path=payload.get("dataset_path"),
            model_config_path=payload.get("model_config_path"),
            event_history=list(payload.get("event_history", [])),
        )
