# -*- coding: utf-8 -*-
"""Pareto frontier utilities and frontier-based reward primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(frozen=True)
class ParetoPoint:
    """A candidate projected into quality/cost space."""

    quality: float
    total_cost: float
    candidate_id: Optional[str] = None
    payload: Optional[Any] = None


def frontier_membership_reward(joined_frontier: bool) -> float:
    """Reward primitive for joining the frontier."""
    if not joined_frontier:
        return 0.0
    return 1.0


def frontier_dominance_bonus(dominated_count: int) -> float:
    """Reward primitive for removing dominated frontier members."""
    if dominated_count < 0:
        raise ValueError("dominated_count must be non-negative")
    return float(dominated_count)


def frontier_contribution_reward(joined_frontier: bool, dominated_count: int = 0) -> float:
    """Combined frontier reward from membership and dominance."""
    return frontier_membership_reward(joined_frontier) + frontier_dominance_bonus(dominated_count)


class ParetoFrontier:
    """Maintains the non-dominated set for (quality, total_cost)."""

    def __init__(self) -> None:
        self._members: List[ParetoPoint] = []

    @property
    def members(self) -> List[ParetoPoint]:
        """Return current frontier members in deterministic order."""
        return list(self._members)

    def add(
        self,
        *,
        quality: float,
        total_cost: float,
        candidate_id: Optional[str] = None,
        payload: Optional[Any] = None,
    ) -> float:
        """
        Add a candidate and return its frontier contribution reward.

        Reward is zero for dominated candidates. For non-dominated additions,
        reward is membership reward plus dominance bonus for removed members.
        """
        point = ParetoPoint(
            quality=quality,
            total_cost=total_cost,
            candidate_id=candidate_id,
            payload=payload,
        )

        if self._is_dominated(point):
            return 0.0

        dominated_members = [member for member in self._members if self._dominates(point, member)]
        if dominated_members:
            self._members = [member for member in self._members if member not in dominated_members]

        self._members.append(point)
        self._members.sort(key=lambda member: (-member.quality, member.total_cost))
        return frontier_contribution_reward(
            joined_frontier=True,
            dominated_count=len(dominated_members),
        )

    def _is_dominated(self, point: ParetoPoint) -> bool:
        """Whether any existing member dominates ``point``."""
        return any(self._dominates(member, point) for member in self._members)

    @staticmethod
    def _dominates(left: ParetoPoint, right: ParetoPoint) -> bool:
        """
        Return true if ``left`` dominates ``right``.

        Dominance means left is no worse on both objectives and better on at least one:
        - quality: higher is better
        - total_cost: lower is better
        """
        return (
            left.quality >= right.quality
            and left.total_cost <= right.total_cost
            and (left.quality > right.quality or left.total_cost < right.total_cost)
        )


__all__ = [
    "ParetoPoint",
    "ParetoFrontier",
    "frontier_membership_reward",
    "frontier_dominance_bonus",
    "frontier_contribution_reward",
]
