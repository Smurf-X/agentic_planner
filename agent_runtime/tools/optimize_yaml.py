# -*- coding: utf-8 -*-
"""YAML optimization tool wrapper."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import yaml  # type: ignore[import-untyped]

from agentic_planner.optimizer.evaluator import StubPipelineEvaluator
from agentic_planner.optimizer.runner import OptimizationRunMode, OptimizationRunner
from agent_runtime.api.schemas import ToolResponse
from agent_runtime.tools.envelope import error_response, ok_response
from agent_runtime.tools.yaml_input import resolve_yaml_input


def optimize_yaml_tool(
    yaml_text_or_path: str,
    objective: str,
    model_config_path: str,
    options: Optional[Mapping[str, Any]],
) -> ToolResponse:
    """Optimize YAML via real MCTS path or fallback stub mode."""
    if not yaml_text_or_path:
        return error_response("missing required argument: yaml_text_or_path")
    if not objective:
        return error_response("missing required argument: objective")
    if not model_config_path and not isinstance(options, Mapping):
        return error_response("missing required argument: model_config_path")

    safe_options = dict(options) if isinstance(options, Mapping) else {}

    yaml_input = resolve_yaml_input(yaml_text_or_path=yaml_text_or_path)
    if not yaml_input.ok:
        return yaml_input
    input_yaml_text = str(yaml_input.data["yaml_text"])

    if not safe_options.get("use_real_optimizer", False):
        return ok_response(
            data={
                "optimized_yaml": input_yaml_text,
                "objective": objective,
                "model_config_path": model_config_path,
                "options": safe_options,
                "candidate_count": 1,
            },
            token_usage=safe_options.get("token_usage"),
        )

    try:
        optimized_yaml, candidate_count, errors = _run_real_optimization(
            yaml_text=input_yaml_text,
            objective=objective,
            options=safe_options,
        )
    except Exception as exc:
        return error_response(f"optimization failed: {exc}")

    return ok_response(
        data={
            "optimized_yaml": optimized_yaml,
            "objective": objective,
            "model_config_path": model_config_path,
            "options": safe_options,
            "candidate_count": candidate_count,
            "errors": errors,
        },
        token_usage=safe_options.get("token_usage"),
    )


def _run_real_optimization(
    *,
    yaml_text: str,
    objective: str,
    options: Dict[str, Any],
) -> tuple[str, int, list[str]]:
    """Run MCTS optimizer with stub evaluator for TUI baseline."""
    cfg = yaml.safe_load(yaml_text)
    if not isinstance(cfg, dict):
        raise ValueError("yaml root must be a mapping")

    max_iterations = int(options.get("max_iterations", 3))
    max_evaluations = int(options.get("max_evaluations", 100))
    mode = str(options.get("optimize_mode", OptimizationRunMode.SEARCH_ONLY.value))

    runner = OptimizationRunner(
        mode=OptimizationRunMode(mode),
        moar_config={
            "max_iterations": max_iterations,
            "max_evaluations": max_evaluations,
            "optimize_goal": objective,
        },
        evaluator=StubPipelineEvaluator(),
    )
    result = runner.run(cfg)
    best_config = result.best_config if result.best_config is not None else cfg
    optimized_yaml = yaml.safe_dump(
        best_config,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    candidate_count = len(result.candidates) if result.candidates else 0
    return optimized_yaml, candidate_count, list(result.errors)
