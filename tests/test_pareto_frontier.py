# -*- coding: utf-8 -*-
"""Tests for Pareto frontier membership and reward primitives."""

from __future__ import annotations

from agentic_planner.optimizer.search.pareto import ParetoFrontier


def test_non_dominated_membership_is_maintained() -> None:
    """Frontier should keep only non-dominated points."""
    frontier = ParetoFrontier()

    frontier.add(quality=0.60, total_cost=10.0, candidate_id="a")
    frontier.add(quality=0.80, total_cost=14.0, candidate_id="b")
    frontier.add(quality=0.70, total_cost=9.0, candidate_id="c")

    members = frontier.members
    assert [m.candidate_id for m in members] == ["b", "c"]


def test_dominated_candidate_gets_zero_reward() -> None:
    """Dominated candidate should not contribute to frontier reward."""
    frontier = ParetoFrontier()

    frontier.add(quality=0.80, total_cost=10.0, candidate_id="a")
    frontier.add(quality=0.70, total_cost=8.0, candidate_id="b")

    reward = frontier.add(quality=0.60, total_cost=12.0, candidate_id="dominated")

    assert reward == 0.0
    assert all(member.candidate_id != "dominated" for member in frontier.members)


def test_new_frontier_member_gets_positive_reward() -> None:
    """Non-dominated additions should produce positive reward."""
    frontier = ParetoFrontier()

    frontier.add(quality=0.80, total_cost=10.0, candidate_id="a")
    frontier.add(quality=0.60, total_cost=6.0, candidate_id="b")

    reward = frontier.add(quality=0.70, total_cost=7.0, candidate_id="c")

    assert reward > 0.0


def test_dominating_candidate_receives_dominance_bonus() -> None:
    """Reward should increase when candidate removes dominated members."""
    frontier = ParetoFrontier()

    frontier.add(quality=0.60, total_cost=10.0, candidate_id="a")
    reward = frontier.add(quality=0.70, total_cost=9.0, candidate_id="b")

    assert reward > 1.0
    assert [m.candidate_id for m in frontier.members] == ["b"]
