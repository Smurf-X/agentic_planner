# -*- coding: utf-8 -*-
"""
Context classes for LLM-guided action selection.

This module provides data structures for tracking action performance
and building rich context for LLM decision making during optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from agentic_planner.contracts.cost import CostBreakdown
    from agentic_planner.contracts.recipe import DJExecutableConfig
    from agentic_planner.optimizer.action import Action


@dataclass
class ActionStats:
    """Statistics for an action type across the search."""

    name: str
    use_count: int = 0
    total_cost_change: float = 0.0
    total_quality_change: float = 0.0

    @property
    def avg_cost_change(self) -> float:
        """Average cost change per use."""
        return self.total_cost_change / self.use_count if self.use_count > 0 else 0.0

    @property
    def avg_quality_change(self) -> float:
        """Average quality change per use."""
        return self.total_quality_change / self.use_count if self.use_count > 0 else 0.0

    def to_summary(self) -> str:
        """Format as summary string for LLM prompt."""
        if self.use_count == 0:
            return f"- {self.name}: 0 uses, avg cost change: Unknown, avg quality change: Unknown"
        return (
            f"- {self.name}: {self.use_count} uses, "
            f"avg cost change: {self.avg_cost_change:+.4f}, "
            f"avg quality change: {self.avg_quality_change:+.4f}"
        )


@dataclass
class TriedAction:
    """Record of a tried action with results."""

    action: Action
    result_quality: float
    result_cost: float
    quality_change: float
    cost_change: float
    applied: bool
    reason: str = ""

    def to_llm_summary(self) -> str:
        """Format for LLM context."""
        status = "+" if self.applied else "x"
        reason_str = f" ({self.reason})" if self.reason else ""
        return (
            f"[{status}] {self.action.directive_name} on {self.action.operator_name}: "
            f"quality {self.quality_change:+.4f}, cost {self.cost_change:+.4f}{reason_str}"
        )


@dataclass
class ActionStatsTracker:
    """
    Tracks action statistics across search iterations.

    Maintains running statistics for each action type, enabling
    LLM to make informed decisions based on historical performance.
    """

    _stats: Dict[str, ActionStats] = field(default_factory=dict)

    def record(
        self,
        action: Action,
        before_quality: float,
        before_cost: float,
        after_quality: float,
        after_cost: float,
    ) -> None:
        """
        Record an action's effect.

        Args:
            action: The action that was applied
            before_quality: Quality before the action
            before_cost: Cost before the action
            after_quality: Quality after the action
            after_cost: Cost after the action
        """
        key = action.directive_name
        if key not in self._stats:
            self._stats[key] = ActionStats(name=key)

        self._stats[key].use_count += 1
        self._stats[key].total_cost_change += after_cost - before_cost
        self._stats[key].total_quality_change += after_quality - before_quality

    def get_stats(self, action_name: str) -> Optional[ActionStats]:
        """Get stats for a specific action."""
        return self._stats.get(action_name)

    def get_all_stats(self) -> Dict[str, ActionStats]:
        """Get all action stats."""
        return self._stats.copy()

    def get_summary(self) -> str:
        """
        Get summary for LLM prompt.

        Returns:
            Formatted string with action statistics
        """
        if not self._stats:
            return "No actions have been tried yet."

        lines = [
            stats.to_summary() for stats in sorted(self._stats.values(), key=lambda s: -s.use_count)
        ]
        return "\n".join(lines)

    def get_top_performers(self, n: int = 5, by: str = "quality") -> List[ActionStats]:
        """
        Get top performing actions.

        Args:
            n: Number of top actions to return
            by: Metric to rank by ("quality" or "cost")

        Returns:
            List of top performing action stats
        """
        if not self._stats:
            return []

        key_func = lambda s: s.avg_quality_change if by == "quality" else -s.avg_cost_change
        sorted_stats = sorted(self._stats.values(), key=key_func, reverse=True)
        return sorted_stats[:n]


@dataclass
class ModelStats:
    """Statistics for a model on the current dataset."""

    model_name: str
    avg_quality: float = 0.0
    avg_cost: float = 0.0
    use_count: int = 0


@dataclass
class ActionSelectionContext:
    """
    Rich context for LLM action selection.

    This class aggregates all information that helps LLM make
    informed decisions about which actions to select.
    """

    config: DJExecutableConfig
    quality: float
    cost: float
    available_actions: List[Action] = field(default_factory=list)

    execution_sample: List[Dict[str, Any]] = field(default_factory=list)
    data_sample: List[Dict[str, Any]] = field(default_factory=list)
    action_stats: ActionStatsTracker = field(default_factory=ActionStatsTracker)
    tried_actions: List[TriedAction] = field(default_factory=list)
    optimize_goal: str = "quality"
    model_stats: Dict[str, ModelStats] = field(default_factory=dict)

    iteration: int = 0
    max_iterations: int = 10

    def get_tried_action_summary(self, limit: int = 10) -> str:
        """Get summary of tried actions for LLM prompt."""
        if not self.tried_actions:
            return "No actions have been tried in this path."

        recent = self.tried_actions[-limit:]
        lines = [ta.to_llm_summary() for ta in recent]
        return "\n".join(lines)

    def get_execution_sample_summary(self, max_chars: int = 2000) -> str:
        """Get formatted execution sample for LLM prompt."""
        import json

        if not self.execution_sample:
            return "No execution results available."

        text = json.dumps(self.execution_sample[:3], indent=2, ensure_ascii=False)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        return text

    def get_data_sample_summary(self, max_chars: int = 2000) -> str:
        """Get formatted data sample for LLM prompt."""
        import json

        if not self.data_sample:
            return "No data sample available."

        text = json.dumps(self.data_sample[:3], indent=2, ensure_ascii=False)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        return text

    def get_model_stats_summary(self) -> str:
        """Get model statistics summary for LLM prompt."""
        if not self.model_stats:
            return "No model statistics available."

        lines = []
        for name, stats in self.model_stats.items():
            lines.append(
                f"- {name}: quality={stats.avg_quality:.4f}, cost=${stats.avg_cost:.2f}, uses={stats.use_count}"
            )
        return "\n".join(lines)


__all__ = [
    "ActionStats",
    "ActionStatsTracker",
    "TriedAction",
    "ModelStats",
    "ActionSelectionContext",
]
