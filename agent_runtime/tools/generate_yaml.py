# -*- coding: utf-8 -*-
"""YAML generation tool wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml  # type: ignore[import-untyped]

from agentic_planner.generator import NLRecipeGenerator, OpenAICompatibleJsonClient
from agentic_planner.optimizer.model_registry import ModelRegistry
from agent_runtime.api.schemas import ToolResponse
from agent_runtime.tools.envelope import error_response, ok_response


def generate_yaml_tool(
    intent: str,
    dataset_path: str,
    model_config_path: str,
    options: Optional[Mapping[str, Any]],
) -> ToolResponse:
    """Generate YAML via real generator path or fallback stub mode."""
    if not intent:
        return error_response("missing required argument: intent")
    if not dataset_path:
        return error_response("missing required argument: dataset_path")

    safe_options = dict(options) if isinstance(options, Mapping) else {}
    use_real_generator = bool(safe_options.get("use_real_generator", False))

    if use_real_generator:
        dataset_candidate = Path(dataset_path)
        if dataset_candidate.is_dir():
            return error_response("dataset_path must be a file path, not a directory")
        if not dataset_candidate.exists():
            return error_response("dataset_path does not exist")

    if not use_real_generator:
        yaml_text = "process:\n  - clean_text_mapper: {}\n"
        return ok_response(
            data={
                "yaml_text": yaml_text,
                "intent": intent,
                "dataset_path": dataset_path,
                "model_config_path": model_config_path,
                "options": safe_options,
            },
            token_usage=safe_options.get("token_usage"),
        )

    try:
        config, token_usage = _run_real_generation(
            intent=intent,
            dataset_path=dataset_path,
            model_config_path=model_config_path,
            options=safe_options,
        )
    except Exception as exc:
        return error_response(f"generation failed: {exc}")

    yaml_text = yaml.safe_dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return ok_response(
        data={
            "yaml_text": yaml_text,
            "intent": intent,
            "dataset_path": dataset_path,
            "model_config_path": model_config_path,
            "options": safe_options,
        },
        token_usage=token_usage,
    )


def _run_real_generation(
    *,
    intent: str,
    dataset_path: str,
    model_config_path: str,
    options: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute real NLRecipeGenerator path."""
    retrieval_mode = str(options.get("retrieval_mode", "none"))
    export_path = str(options.get("export_path", _default_export_path(dataset_path)))
    candidate_top_k = int(options.get("candidate_top_k", 20))

    llm_client = _build_llm_client(model_config_path=model_config_path, options=options)
    try:
        generator = NLRecipeGenerator(llm=llm_client, retrieval_mode=retrieval_mode)
        config = generator.generate(
            user_intent=intent,
            dataset_path=dataset_path,
            export_path=export_path,
            dataset_hint=str(options.get("dataset_hint", "")),
            candidate_top_k=candidate_top_k,
        )
        return config, llm_client.get_last_usage()
    finally:
        llm_client.close()


def _default_export_path(dataset_path: str) -> str:
    """Build default export path near dataset."""
    p = Path(dataset_path)
    stem = p.stem or "output"
    return str(p.with_name(f"{stem}_processed.jsonl"))


def _build_llm_client(*, model_config_path: str, options: Dict[str, Any]) -> OpenAICompatibleJsonClient:
    """Create LLM client from direct settings, model config, or environment."""
    if options.get("model"):
        return OpenAICompatibleJsonClient(
            model=str(options.get("model")),
            api_key=str(options.get("api_key") or ""),
            base_url=str(options.get("base_url") or ""),
        )

    if model_config_path:
        registry = ModelRegistry.from_yaml(model_config_path)
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

    fallback_model = str(options.get("fallback_model", "gpt-4o-mini"))
    return OpenAICompatibleJsonClient(model=fallback_model)
