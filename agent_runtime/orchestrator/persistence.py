# -*- coding: utf-8 -*-
"""Local persistence for runtime session state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from agent_runtime.orchestrator.session_state import SessionState


class SessionPersistence:
    """Persist session state to local JSON files."""

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize storage directory for session snapshots."""
        if storage_dir is None:
            self._storage_dir = Path(".agent_runtime") / "sessions"
        else:
            self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Build deterministic path for one session id."""
        return self._storage_dir / f"{session_id}.json"

    def save(self, state: SessionState) -> str:
        """Save one session snapshot and return the path."""
        path = self._session_path(state.session_id)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state.to_dict(), handle, sort_keys=True)
        return str(path)

    def load(self, session_id: str) -> SessionState:
        """Load one session snapshot by id."""
        path = self._session_path(session_id)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return SessionState.from_dict(payload)
