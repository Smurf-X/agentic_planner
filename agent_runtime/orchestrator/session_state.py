# -*- coding: utf-8 -*-
"""Session state for agent runtime orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _coerce_str_or_default(value: Any, default: str = "") -> str:
    """Return string values or a safe default."""
    if isinstance(value, str):
        return value
    return default


def _coerce_optional_str(value: Any) -> Optional[str]:
    """Return a string value when valid, else None."""
    if isinstance(value, str):
        return value
    return None


def _coerce_dict_list(value: Any) -> List[Dict[str, Any]]:
    """Keep only dictionary entries from list-like payloads."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


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
        if not isinstance(payload, dict):
            raise ValueError("session snapshot payload must be a mapping")

        return cls(
            session_id=_coerce_str_or_default(payload.get("session_id", "")),
            current_yaml=_coerce_str_or_default(payload.get("current_yaml", "")),
            last_generated_yaml=_coerce_str_or_default(payload.get("last_generated_yaml", "")),
            last_optimized_candidates=_coerce_dict_list(payload.get("last_optimized_candidates", [])),
            objective=_coerce_optional_str(payload.get("objective")),
            dataset_path=_coerce_optional_str(payload.get("dataset_path")),
            model_config_path=_coerce_optional_str(payload.get("model_config_path")),
            event_history=_coerce_dict_list(payload.get("event_history", [])),
        )
