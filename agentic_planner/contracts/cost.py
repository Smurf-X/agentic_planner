# -*- coding: utf-8 -*-
"""Cost model: LLM token money cost + wall time (independent dimensions)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping


@dataclass
class CostBreakdown:
    """Dual cost dimensions for a candidate configuration run."""

    llm_token_cost: float = 0.0
    """Monetary or normalized cost from prompt/completion tokens × price table."""

    wall_time_sec: float = 0.0
    """End-to-end execution time on the evaluation sample."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_usage: Dict[str, Dict[str, int]] = field(default_factory=dict)
    """Optional per-model token counts, e.g. ``{"gpt-4o-mini": {"prompt": 10, "completion": 5}}``."""

    def __repr__(self) -> str:
        return (
            f"CostBreakdown(llm_token_cost={self.llm_token_cost:.6g}, "
            f"wall_time_sec={self.wall_time_sec:.4f}, "
            f"tokens={self.prompt_tokens}+{self.completion_tokens})"
        )


def compute_token_cost(
    usage: Mapping[str, Mapping[str, int]],
    price_per_million: Mapping[str, float],
) -> float:
    """
    Sum ``(prompt + completion) * price / 1e6`` for each model key present in ``usage``.

    ``usage`` values are dicts with optional keys ``prompt`` and ``completion`` (ints).
    Missing models in ``price_per_million`` contribute ``0`` and should be logged by callers.
    """
    total = 0.0
    for model, counts in usage.items():
        price = float(price_per_million.get(model, 0.0))
        if price == 0.0:
            continue
        p = int(counts.get("prompt", 0) or 0)
        c = int(counts.get("completion", 0) or 0)
        total += (p + c) / 1_000_000.0 * price
    return total