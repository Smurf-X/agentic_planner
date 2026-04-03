# -*- coding: utf-8 -*-
"""JSONL event logger for runtime actions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, Union


class EventLogger:
    """Append deterministic runtime events to a JSONL file."""

    _REQUIRED_FIELDS = (
        "timestamp",
        "tool_name",
        "input_summary",
        "result_summary",
        "duration_ms",
        "token_usage",
        "error",
    )

    def __init__(self, log_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize runtime log directory and default event file."""
        self._log_dir = Path(log_dir) if log_dir is not None else Path(".agent_runtime") / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._log_dir / "events.jsonl"

    @staticmethod
    def now_timestamp() -> str:
        """Return current UTC timestamp in ISO-8601 format."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def now_counter() -> float:
        """Return monotonic clock value for duration tracking."""
        return perf_counter()

    @staticmethod
    def duration_ms(start_counter: float) -> int:
        """Return elapsed milliseconds from a monotonic start value."""
        elapsed = int((perf_counter() - start_counter) * 1000)
        if elapsed < 0:
            return 0
        return elapsed

    @property
    def log_path(self) -> Path:
        """Return the current JSONL file path."""
        return self._log_path

    def _normalize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure required event schema keys exist."""
        normalized: Dict[str, Any] = dict(event)
        if "timestamp" not in normalized:
            normalized["timestamp"] = self.now_timestamp()
        if "tool_name" not in normalized:
            normalized["tool_name"] = "unknown"
        if "input_summary" not in normalized:
            normalized["input_summary"] = {}
        if "result_summary" not in normalized:
            normalized["result_summary"] = {}
        if "duration_ms" not in normalized:
            normalized["duration_ms"] = 0
        if "token_usage" not in normalized:
            normalized["token_usage"] = None
        if "error" not in normalized:
            normalized["error"] = None

        return {field: normalized[field] for field in self._REQUIRED_FIELDS}

    def log_event(self, event: Dict[str, Any]) -> None:
        """Append one normalized event to the JSONL stream."""
        normalized_event = self._normalize_event(event)
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(normalized_event, sort_keys=True))
            handle.write("\n")
