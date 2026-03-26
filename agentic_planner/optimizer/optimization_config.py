# -*- coding: utf-8 -*-
"""Optimization configuration management.

This module provides:
1. OptimizationConfig - Full configuration for optimization runs
2. PriceTable - Token pricing configuration
3. Configuration loading and validation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from agentic_planner.contracts.eval_protocol import EvalConfig, EvaluationMode
from agentic_planner.optimizer.directive_engine import DirectiveEngineConfig, DirectiveEngineMode
from agentic_planner.optimizer.search.moar import MOARSearchConfig


class PriceTable(BaseModel):
    """Token pricing configuration per model (USD per million tokens)."""

    prices: Dict[str, float] = Field(
        default_factory=lambda: {
            "gpt-4o-mini": 0.15,
            "gpt-4o": 2.5,
            "gpt-4-turbo": 10.0,
            "gpt-3.5-turbo": 0.5,
            "claude-3-haiku": 0.25,
            "claude-3-sonnet": 3.0,
            "claude-3-opus": 15.0,
        },
        description="Price per million tokens per model.",
    )

    def get_price(self, model: str) -> float:
        """Get price for a model, with fallback to 0."""
        return self.prices.get(model, 0.0)

    def to_dict(self) -> Dict[str, float]:
        """Convert to plain dict."""
        return dict(self.prices)


class LLMConfig(BaseModel):
    """LLM client configuration."""

    model: str = Field(default="gpt-4o-mini", description="Model to use for inference/judging")
    api_base: Optional[str] = Field(default=None, description="API base URL (for custom endpoints)")
    api_key_env: str = Field(default="OPENAI_API_KEY", description="Environment variable for API key")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)


class ExecutionConfig(BaseModel):
    """Pipeline execution configuration."""

    dataset_path: Optional[str] = Field(default=None, description="Path to input dataset (JSONL)")
    ground_truth_path: Optional[str] = Field(default=None, description="Path to ground truth labels")
    work_dir: str = Field(default="/tmp/dj_optimizer", description="Working directory for temp files")
    num_workers: int = Field(default=4, ge=1, description="Number of parallel workers")


class SearchExecutionBoundaryConfig(BaseModel):
    """Runtime boundary settings enforced during search evaluations."""

    disable_op_fusion: bool = Field(
        default=True,
        description="Disable DJ op_fusion so evaluated configs remain explicit.",
    )
    disable_checkpoint_optimization: bool = Field(
        default=True,
        description="Disable DJ checkpoint optimization during search evaluation.",
    )
    disable_partition_optimization: bool = Field(
        default=True,
        description="Disable DJ partition optimization during search evaluation.",
    )

    def to_runtime_overrides(self) -> Dict[str, bool]:
        """Build DJ runtime overrides for search evaluations."""
        return {
            "op_fusion": not self.disable_op_fusion,
            "checkpoint_optimization": not self.disable_checkpoint_optimization,
            "partition_optimization": not self.disable_partition_optimization,
        }


class OptimizationConfig(BaseModel):
    """
    Full configuration for optimization runs.

    This is the main configuration class that covers all aspects of
    the optimization pipeline, including directives, search, evaluation,
    and execution settings.
    """

    # Run mode
    run_mode: str = Field(
        default="directive_only",
        description="Mode: directive_only, search_only, or directive_then_search",
    )

    # Stage 1: Directive engine
    directive: DirectiveEngineConfig = Field(
        default_factory=DirectiveEngineConfig,
        description="Directive engine configuration",
    )

    # Stage 2: Search (optional)
    search: Optional[MOARSearchConfig] = Field(
        default=None,
        description="MOAR search configuration (for search mode)",
    )

    # Evaluation
    evaluation: EvalConfig = Field(
        default_factory=EvalConfig,
        description="Evaluation configuration",
    )

    # Execution
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Pipeline execution configuration",
    )

    # Search runtime boundary
    search_execution_boundary: SearchExecutionBoundaryConfig = Field(
        default_factory=SearchExecutionBoundaryConfig,
        description="Execution boundary settings enforced for search runs",
    )

    # LLM settings
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM client configuration",
    )

    # Pricing
    pricing: PriceTable = Field(
        default_factory=PriceTable,
        description="Token pricing table",
    )

    # Output settings
    output_dir: Optional[str] = Field(
        default=None,
        description="Directory to save optimized configs",
    )
    save_trace: bool = Field(
        default=True,
        description="Save optimization trace",
    )

    model_config = {"extra": "allow"}

    @field_validator("run_mode")
    @classmethod
    def validate_run_mode(cls, v: str) -> str:
        valid = {"directive_only", "search_only", "directive_then_search"}
        if v not in valid:
            raise ValueError(f"run_mode must be one of {valid}")
        return v

    @model_validator(mode="after")
    def validate_consistency(self) -> "OptimizationConfig":
        """Validate configuration consistency."""
        # If search mode, ensure search config is provided
        if self.run_mode in ("search_only", "directive_then_search"):
            if self.search is None:
                # Create default search config
                self.search = MOARSearchConfig()
        return self

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        return yaml.safe_dump(
            self.model_dump(mode="python"),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "OptimizationConfig":
        """Parse from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: str | Path) -> "OptimizationConfig":
        """Load from YAML file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        return cls.from_yaml(p.read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        """Save to YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml(), encoding="utf-8")


# Default configuration templates


DEFAULT_DIRECTIVE_ONLY_CONFIG = OptimizationConfig(
    run_mode="directive_only",
    directive=DirectiveEngineConfig(
        mode=DirectiveEngineMode.STATIC,
        directives=["reorder_filters_first", "remove_redundant_ops"],
    ),
    evaluation=EvalConfig(
        mode=EvaluationMode.NO_LABELS,
        sample_size=50,
    ),
)

DEFAULT_INFERENCE_CONFIG = OptimizationConfig(
    run_mode="directive_only",
    directive=DirectiveEngineConfig(
        mode=DirectiveEngineMode.INFERENCE,
        task_description="",
        optimization_goal="balance cost and quality",
    ),
    evaluation=EvalConfig(
        mode=EvaluationMode.NO_LABELS,
        sample_size=50,
    ),
)

DEFAULT_SEARCH_CONFIG = OptimizationConfig(
    run_mode="search_only",
    search=MOARSearchConfig(
        max_iterations=3,
    ),
    evaluation=EvalConfig(
        mode=EvaluationMode.NO_LABELS,
        sample_size=50,
    ),
)

DEFAULT_FULL_CONFIG = OptimizationConfig(
    run_mode="directive_then_search",
    directive=DirectiveEngineConfig(
        mode=DirectiveEngineMode.STATIC,
        directives=["reorder_filters_first", "remove_redundant_ops"],
    ),
    search=MOARSearchConfig(
        max_iterations=3,
    ),
    evaluation=EvalConfig(
        mode=EvaluationMode.NO_LABELS,
        sample_size=50,
    ),
)


# Configuration file template (for user reference)

CONFIG_TEMPLATE = """# Data-Juicer Pipeline Optimization Configuration
# Save this file and customize for your use case

# Run mode: directive_only, search_only, or directive_then_search
run_mode: directive_only

# Stage 1: Directive-based optimization
directive:
  # Mode: static (use predefined list) or inference (use LLM to infer)
  mode: static
  
  # List of directives to apply (for static mode)
  # Available: reorder_filters_first, remove_redundant_ops, 
  #            tighten_filters, loosen_filters, bump_text_length_min_len
  directives:
    - reorder_filters_first
    - remove_redundant_ops
  
  # Maximum steps to apply
  max_steps: 50
  
  # For inference mode:
  # task_description: "Process text data for training"
  # optimization_goal: "balance cost and quality"

# Stage 2: Search-based optimization (optional)
# search:
#   max_iterations: 3
#   max_evaluations: 100
#   exploration_weight: 1.4

# Evaluation settings
evaluation:
  # Mode: no_labels (LLM judge) or with_ground_truth
  mode: no_labels
  sample_size: 50
  random_seed: 42
  judge_model: gpt-4o-mini
  task_description: ""

# Execution settings
execution:
  dataset_path: null  # Set to your dataset path
  ground_truth_path: null
  work_dir: /tmp/dj_optimizer
  num_workers: 4

# Search execution boundary (applies to search modes)
search_execution_boundary:
  disable_op_fusion: true
  disable_checkpoint_optimization: true
  disable_partition_optimization: true

# LLM configuration
llm:
  model: gpt-4o-mini
  api_base: null  # For custom endpoints
  api_key_env: OPENAI_API_KEY
  temperature: 0.3
  max_tokens: 4096

# Token pricing (USD per million tokens)
pricing:
  prices:
    gpt-4o-mini: 0.15
    gpt-4o: 2.5
    gpt-3.5-turbo: 0.5
    claude-3-haiku: 0.25
    claude-3-sonnet: 3.0

# Output settings
output_dir: null  # Directory to save optimized configs
save_trace: true
"""


def create_sample_config_file(path: str | Path) -> None:
    """Create a sample configuration file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(CONFIG_TEMPLATE, encoding="utf-8")


def load_config(
    path: Optional[str | Path] = None,
    **overrides: Any,
) -> OptimizationConfig:
    """
    Load configuration with optional overrides.

    Args:
        path: Path to config file (optional)
        **overrides: Override specific fields

    Returns:
        OptimizationConfig instance
    """
    if path:
        config = OptimizationConfig.from_file(path)
    else:
        config = OptimizationConfig()

    # Apply overrides
    if overrides:
        data = config.model_dump()
        data.update(overrides)
        config = OptimizationConfig.model_validate(data)

    return config
