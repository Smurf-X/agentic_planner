# -*- coding: utf-8 -*-
"""YAML optimization tool wrapper."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import yaml

from agentic_planner.generator import OpenAICompatibleJsonClient
from agentic_planner.optimizer.evaluator import StubPipelineEvaluator
from agentic_planner.optimizer.runner import OptimizationRunMode, OptimizationRunner
from agent_runtime.api.schemas import ToolResponse
from agent_runtime.tools.envelope import error_response, ok_response
from agent_runtime.tools.yaml_input import resolve_yaml_input


def optimize_yaml_tool(
    yaml_text_or_path: str,
    objective: str,
    llm_config_path: str,
    options: Optional[Mapping[str, Any]],
    payload: Optional[Mapping[str, Any]] = None,
) -> ToolResponse:
    """Optimize YAML via real MCTS path or fallback stub mode."""
    if not yaml_text_or_path:
        return error_response("missing required argument: yaml_text_or_path")
    if not objective:
        return error_response("missing required argument: objective")

    safe_options = dict(options) if isinstance(options, Mapping) else {}
    safe_payload = dict(payload) if isinstance(payload, Mapping) else {}

    yaml_input = resolve_yaml_input(yaml_text_or_path=yaml_text_or_path)
    if not yaml_input.ok:
        return yaml_input
    input_yaml_text = str(yaml_input.data["yaml_text"])

    llm_config = safe_payload.get("llm_config")
    has_llm_config = llm_config and llm_config.get("model") and llm_config.get("api_key")

    if not safe_options.get("use_real_optimizer", False):
        return ok_response(
            data={
                "optimized_yaml": input_yaml_text,
                "objective": objective,
                "llm_config_path": llm_config_path,
                "options": safe_options,
                "candidate_count": 1,
            },
            token_usage=safe_options.get("token_usage"),
        )

    if not has_llm_config and not llm_config_path:
        return error_response("Please provide LLM credentials (base_url, api_key, model) in the form")

    try:
        optimized_yaml, candidate_count, errors = _run_real_optimization(
            yaml_text=input_yaml_text,
            objective=objective,
            options=safe_options,
            llm_config=llm_config,
            llm_config_path=llm_config_path,
        )
    except Exception as exc:
        return error_response(f"optimization failed: {exc}")

    return ok_response(
        data={
            "optimized_yaml": optimized_yaml,
            "objective": objective,
            "llm_config_path": llm_config_path,
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
    llm_config: Optional[Dict[str, Any]] = None,
    llm_config_path: str = "",
) -> tuple[str, int, list[str]]:
    """Run MCTS optimizer with stub evaluator for TUI baseline."""
    cfg = yaml.safe_load(yaml_text)
    if not isinstance(cfg, dict):
        raise ValueError("yaml root must be a mapping")

    max_iterations = int(options.get("max_iterations", 3))
    max_evaluations = int(options.get("max_evaluations", 100))
    mode = str(options.get("optimize_mode", OptimizationRunMode.SEARCH_ONLY.value))

    llm_client = _build_llm_client(
        llm_config_path=llm_config_path,
        options=options,
        llm_config=llm_config,
    )

    try:
        runner = OptimizationRunner(
            mode=OptimizationRunMode(mode),
            moar_config={
                "max_iterations": max_iterations,
                "max_evaluations": max_evaluations,
                "optimize_goal": objective,
            },
            evaluator=StubPipelineEvaluator(),
            llm_client=llm_client,
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
    finally:
        llm_client.close()


def _build_llm_client(
    *,
    llm_config_path: str,
    options: Dict[str, Any],
    llm_config: Optional[Dict[str, Any]] = None,
) -> OpenAICompatibleJsonClient:
    """Create LLM client from llm_config, direct settings, or model config file."""
    if llm_config:
        base_url = llm_config.get("base_url", "")
        api_key = llm_config.get("api_key", "")
        model = llm_config.get("model", "")
        if model and api_key:
            return OpenAICompatibleJsonClient(
                model=model,
                api_key=api_key,
                base_url=base_url or "https://api.openai.com/v1",
            )

    if options.get("model"):
        return OpenAICompatibleJsonClient(
            model=str(options.get("model")),
            api_key=str(options.get("api_key") or ""),
            base_url=str(options.get("base_url") or ""),
        )

    if llm_config_path:
        from agentic_planner.optimizer.model_registry import ModelRegistry
        registry = ModelRegistry.from_yaml(llm_config_path)
        preferred_model = str(options.get("preferred_model") or "")
        if preferred_model:
            return registry.create_client(preferred_model)

        candidates = registry.get_candidate_models()
        if candidates:
            return registry.create_client(candidates[0])

        names = registry.list_models()
        if not names:
            raise ValueError("no models available in model config")
        return registry.create_client(names[0])

    raise ValueError("Please provide LLM credentials (base_url, api_key, model) in the form")