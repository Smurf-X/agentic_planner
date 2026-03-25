# -*- coding: utf-8 -*-
"""Bridge between internal plan operator list and DJ ``process`` format."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableSequence


@dataclass
class OperatorStep:
    """One operator as used inside NL generation (agents-style intermediate)."""

    name: str
    params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OperatorStep":
        name = str(data.get("name", "")).strip()
        params = data.get("params", {})
        if not isinstance(params, dict):
            params = {}
        return cls(name=name, params=params)


PlanOperators = List[OperatorStep]


def plan_operators_to_process(operators: PlanOperators) -> List[Dict[str, Any]]:
    """
    Convert ``[{name, params}, ...]`` to DJ executable ``process`` list:

    ``[ {op_name: params}, ... ]``.
    """
    out: List[Dict[str, Any]] = []
    for op in operators:
        if not op.name:
            continue
        out.append({op.name: dict(op.params)})
    return out


def process_to_plan_operators(process: MutableSequence[Mapping[str, Any]]) -> PlanOperators:
    """Inverse of :func:`plan_operators_to_process` (for round-trips / debugging)."""
    steps: List[OperatorStep] = []
    for step in process:
        if not isinstance(step, dict) or len(step) != 1:
            continue
        name, params = next(iter(step.items()))
        if not isinstance(params, dict):
            params = {}
        steps.append(OperatorStep(name=str(name), params=dict(params)))
    return steps