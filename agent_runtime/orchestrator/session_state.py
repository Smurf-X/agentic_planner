# -*- coding: utf-8 -*-
"""Session state for agent runtime orchestration."""

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
