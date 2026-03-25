# -*- coding: utf-8 -*-
"""Evaluation protocol types (sample-based judge + optional ground truth)."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class EvaluationMode(str, Enum):
    """How quality is computed for a candidate pipeline."""

    NO_LABELS = "no_labels"
    """LLM-as-judge on per-sample outputs."""

    WITH_GROUND_TRUTH = "with_ground_truth"
    """Compare model outputs to reference field (path configurable)."""


class EvalConfig(BaseModel):
    """Configuration for the sample-based evaluator (v1)."""

    mode: EvaluationMode = Field(
        default=EvaluationMode.NO_LABELS,
        description="no_labels: judge only; with_ground_truth: compare to reference column",
    )
    sample_size: int = Field(default=50, ge=1, le=10_000)
    random_seed: int = Field(default=42)
    judge_model: Optional[str] = Field(
        default=None,
        description="Model id for LLM-as-judge when mode is NO_LABELS.",
    )
    judge_prompt_template: Optional[str] = Field(
        default=None,
        description="Optional Jinja-style or plain template; implementation may ignore in v1.",
    )
    ground_truth_key: Optional[str] = Field(
        default=None,
        description="Dataset field name for reference labels when mode is WITH_GROUND_TRUTH.",
    )
    task_description: str = Field(
        default="",
        description="User-facing task text passed to the judge.",
    )

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def _require_ground_truth_when_needed(self) -> "EvalConfig":
        if self.mode == EvaluationMode.WITH_GROUND_TRUTH and not self.ground_truth_key:
            raise ValueError("ground_truth_key is required when mode is with_ground_truth")
        return self