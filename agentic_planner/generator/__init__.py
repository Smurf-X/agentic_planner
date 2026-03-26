# -*- coding: utf-8 -*-
"""
Generator module - Natural language to Data-Juicer YAML configuration.

This module provides:
- NLRecipeGenerator: Main class for generating pipelines from natural language
- Embedding backends for vector retrieval
- Candidate filtering and ranking
"""

from agentic_planner.generator.llm import (
    LLMJsonClient,
    DictLLMJsonClient,
    parse_json_object_strict,
)
from agentic_planner.generator.http_llm import OpenAICompatibleJsonClient
from agentic_planner.generator.catalog import (
    build_operator_catalog_text,
    build_operator_detail_text,
)
from agentic_planner.generator.op_schema import (
    build_schema_block,
    format_allowlist_for_prompt,
    get_init_param_allowlist,
    sanitize_params,
    validate_params_bind,
)
from agentic_planner.generator.generator import (
    NLRecipeGenerator,
    assemble_executable_config,
    generate_recipe_from_llm_json_text,
)

__all__ = [
    # LLM clients
    "LLMJsonClient",
    "DictLLMJsonClient",
    "parse_json_object_strict",
    "OpenAICompatibleJsonClient",
    # Catalog
    "build_operator_catalog_text",
    "build_operator_detail_text",
    # Schema
    "build_schema_block",
    "format_allowlist_for_prompt",
    "get_init_param_allowlist",
    "sanitize_params",
    "validate_params_bind",
    # Main generator
    "NLRecipeGenerator",
    "assemble_executable_config",
    "generate_recipe_from_llm_json_text",
]