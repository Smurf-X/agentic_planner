# -*- coding: utf-8 -*-
"""Stage-1: apply a list of directives in order with an audit trail.

Supports two modes:
1. Static mode: Apply a pre-configured list of directives
2. Inference mode: Use LLM to infer which directives to apply
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agentic_planner.contracts.recipe import DJExecutableConfig, validate_executable_config
from agentic_planner.optimizer.directives.base import DirectiveResult
from agentic_planner.optimizer.directives.registry import DIRECTIVE_REGISTRY

if TYPE_CHECKING:
    pass


class DirectiveEngineMode(str, Enum):
    """How directives are determined."""

    STATIC = "static"
    """Use pre-configured directive list."""

    INFERENCE = "inference"
    """Use LLM to infer directives."""


class DirectiveEngineConfig(BaseModel):
    """YAML-friendly config for DirectiveEngine."""

    mode: DirectiveEngineMode = Field(
        default=DirectiveEngineMode.STATIC,
        description="static: use configured directives list; inference: use LLM to infer",
    )
    directives: List[str] = Field(
        default_factory=list,
        description="Directive keys to apply in order (see DIRECTIVE_REGISTRY). Used in static mode.",
    )
    max_steps: int = Field(default=50, ge=1, le=500)
    """Max applied directives (same name may repeat if listed)."""

    task_description: str = Field(
        default="",
        description="Description of the data processing task. Used in inference mode.",
    )
    optimization_goal: str = Field(
        default="balance cost and quality",
        description="Optimization objective. Used in inference mode.",
    )

    model_config = {"extra": "allow"}


@dataclass
class DirectiveEngineRun:
    """Result of a directive-only pass."""

    ok: bool
    config: DJExecutableConfig
    trace: List[DirectiveResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    mode: DirectiveEngineMode = DirectiveEngineMode.STATIC
    recommendations: Optional[Any] = None  # DirectiveRecommendations when in inference mode


class DirectiveEngine:
    """
    Apply configured directives to transform a pipeline configuration.

    Supports two modes:
    - Static: Apply directives in the order specified in config
    - Inference: Use LLM to analyze the pipeline and recommend directives
    """

    def __init__(
        self,
        config: DirectiveEngineConfig,
        llm_client: Optional[Any] = None,
    ) -> None:
        """
        Args:
            config: Engine configuration
            llm_client: LLM client for inference mode (optional for static mode)
        """
        self._config = config
        self._llm_client = llm_client

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        llm_client: Optional[Any] = None,
    ) -> "DirectiveEngine":
        """Create engine from configuration dict."""
        return cls(DirectiveEngineConfig.model_validate(data), llm_client)

    def set_llm_client(self, client: Any) -> None:
        """Set the LLM client for inference mode."""
        self._llm_client = client

    def run(self, cfg: DJExecutableConfig) -> DirectiveEngineRun:
        """
        Run the directive engine on the given configuration.

        In static mode, applies configured directives in order.
        In inference mode, uses LLM to determine which directives to apply.
        """
        if self._config.mode == DirectiveEngineMode.INFERENCE:
            return self._run_inference_mode(cfg)
        return self._run_static_mode(cfg)

    def _run_static_mode(self, cfg: DJExecutableConfig) -> DirectiveEngineRun:
        """Apply directives in the configured order."""
        trace: List[DirectiveResult] = []
        errors: List[str] = []
        current = cfg
        steps = 0

        for name in self._config.directives:
            if steps >= self._config.max_steps:
                break
            directive = DIRECTIVE_REGISTRY.get(name)
            if directive is None:
                errors.append(f"unknown directive: {name}")
                continue
            res = directive.apply(current)
            trace.append(res)
            if not res.ok:
                errors.append(f"{name}: directive failed")
                break
            if res.config_after is not None:
                current = res.config_after
            steps += 1

        val_err = validate_executable_config(current)
        if val_err:
            errors.extend(val_err)

        return DirectiveEngineRun(
            ok=not errors,
            config=current,
            trace=trace,
            errors=errors,
            mode=DirectiveEngineMode.STATIC,
        )

    def _run_inference_mode(self, cfg: DJExecutableConfig) -> DirectiveEngineRun:
        """Use LLM to infer and apply directives."""
        # Import here to avoid circular dependency
        try:
            from agentic_planner.optimizer.directive_inference import DirectiveInferenceEngine
        except ImportError:
            # Fallback to static mode if inference module not available
            return self._run_static_mode(cfg)

        inference_engine = DirectiveInferenceEngine(
            llm_client=self._llm_client,
            task_description=self._config.task_description,
            optimization_goal=self._config.optimization_goal,
            max_directives=self._config.max_steps,
        )

        result = inference_engine.run(cfg)

        return DirectiveEngineRun(
            ok=result.ok,
            config=result.config,
            trace=result.trace,
            errors=result.errors,
            mode=DirectiveEngineMode.INFERENCE,
            recommendations=result.recommendations,
        )


# Convenience functions for common use cases


def apply_static_directives(
    cfg: DJExecutableConfig,
    directives: List[str],
) -> DirectiveEngineRun:
    """
    Apply a list of directives to a configuration.

    Args:
        cfg: The pipeline configuration
        directives: List of directive names to apply

    Returns:
        DirectiveEngineRun with the transformed config
    """
    config = DirectiveEngineConfig(
        mode=DirectiveEngineMode.STATIC,
        directives=directives,
    )
    engine = DirectiveEngine(config)
    return engine.run(cfg)


def optimize_with_inference(
    cfg: DJExecutableConfig,
    llm_client: Any,
    task_description: str = "",
    optimization_goal: str = "balance cost and quality",
) -> DirectiveEngineRun:
    """
    Use LLM to infer and apply optimal directives.

    Args:
        cfg: The pipeline configuration
        llm_client: LLM client for inference
        task_description: Description of the task
        optimization_goal: What to optimize for

    Returns:
        DirectiveEngineRun with recommendations and transformed config
    """
    config = DirectiveEngineConfig(
        mode=DirectiveEngineMode.INFERENCE,
        task_description=task_description,
        optimization_goal=optimization_goal,
    )
    engine = DirectiveEngine(config, llm_client)
    return engine.run(cfg)


__all__ = [
    "DirectiveEngine",
    "DirectiveEngineConfig",
    "DirectiveEngineMode",
    "DirectiveEngineRun",
    "apply_static_directives",
    "optimize_with_inference",
]
