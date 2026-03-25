# -*- coding: utf-8 -*-
"""
Agentic Planner - Natural language pipeline design and optimization.

This package provides:
1. Generator: Convert natural language to Data-Juicer YAML configs
2. Optimizer: Optimize pipeline configurations using directives and search

Usage:
    from agentic_planner import PipelineGenerator, PipelineOptimizer
    
    # Generate from natural language
    generator = PipelineGenerator(llm_client=openai_client)
    config = generator.generate("Filter short texts and extract keywords")
    
    # Optimize the config
    optimizer = PipelineOptimizer(mode="directive_then_search")
    result = optimizer.run(config)
"""

__version__ = "0.1.0"

# Contracts
from agentic_planner.contracts.recipe import (
    DJExecutableConfig,
    load_executable_config,
    save_executable_config,
    validate_executable_config,
)
from agentic_planner.contracts.cost import (
    CostBreakdown,
    compute_token_cost,
)
from agentic_planner.contracts.eval_protocol import (
    EvalConfig,
    EvaluationMode,
)
from agentic_planner.contracts.plan_bridge import (
    OperatorStep,
    plan_operators_to_process,
    process_to_plan_operators,
)

# Generator
from agentic_planner.generator import (
    NLRecipeGenerator,
    OpenAICompatibleJsonClient,
    assemble_executable_config,
    build_operator_catalog_text,
    build_operator_detail_text,
    generate_recipe_from_llm_json_text,
)

# Optimizer - Core
from agentic_planner.optimizer.op_locator import (
    OpIdentity,
    OpLocator,
    ProcessIndex,
)
from agentic_planner.optimizer.directive_engine import (
    DirectiveEngine,
    DirectiveEngineConfig,
    DirectiveEngineMode,
    DirectiveEngineRun,
    apply_static_directives,
)
from agentic_planner.optimizer.optimization_config import (
    OptimizationConfig,
    ExecutionConfig,
    LLMConfig,
    PriceTable,
    load_config,
)
from agentic_planner.optimizer.runner import (
    OptimizationRunner,
    OptimizationRunnerResult,
    OptimizationRunMode,
)

# Optimizer - Directives
from agentic_planner.optimizer.directives import (
    Directive,
    DirectiveResult,
    DIRECTIVE_REGISTRY,
    register_directive,
    get_directive,
    list_directive_names,
    ReorderFiltersFirstDirective,
    RemoveRedundantOpDirective,
    TightenFiltersDirective,
    LoosenFiltersDirective,
    BumpMinLenDirective,
)

# Optimizer - Search
from agentic_planner.optimizer.search import (
    BaseSearchStrategy,
    OptimizationObjective,
    SearchConfig,
    SearchReport,
    SearchResult,
    SearchStrategyType,
    GreedySearchConfig,
    GreedySearchStrategy,
    RandomSearchConfig,
    RandomSearchStrategy,
    BeamSearchConfig,
    BeamSearchStrategy,
    create_search_strategy,
)

__all__ = [
    # Version
    "__version__",
    # Contracts
    "DJExecutableConfig",
    "load_executable_config",
    "save_executable_config",
    "validate_executable_config",
    "CostBreakdown",
    "compute_token_cost",
    "EvalConfig",
    "EvaluationMode",
    "OperatorStep",
    "plan_operators_to_process",
    "process_to_plan_operators",
    # Generator
    "NLRecipeGenerator",
    "OpenAICompatibleJsonClient",
    "assemble_executable_config",
    "build_operator_catalog_text",
    "build_operator_detail_text",
    "generate_recipe_from_llm_json_text",
    # Optimizer - Core
    "OpIdentity",
    "OpLocator",
    "ProcessIndex",
    "DirectiveEngine",
    "DirectiveEngineConfig",
    "DirectiveEngineMode",
    "DirectiveEngineRun",
    "apply_static_directives",
    "OptimizationConfig",
    "ExecutionConfig",
    "LLMConfig",
    "PriceTable",
    "load_config",
    "OptimizationRunner",
    "OptimizationRunnerResult",
    "OptimizationRunMode",
    # Optimizer - Directives
    "Directive",
    "DirectiveResult",
    "DIRECTIVE_REGISTRY",
    "register_directive",
    "get_directive",
    "list_directive_names",
    "ReorderFiltersFirstDirective",
    "RemoveRedundantOpDirective",
    "TightenFiltersDirective",
    "LoosenFiltersDirective",
    "BumpMinLenDirective",
    # Optimizer - Search
    "BaseSearchStrategy",
    "OptimizationObjective",
    "SearchConfig",
    "SearchReport",
    "SearchResult",
    "SearchStrategyType",
    "GreedySearchConfig",
    "GreedySearchStrategy",
    "RandomSearchConfig",
    "RandomSearchStrategy",
    "BeamSearchConfig",
    "BeamSearchStrategy",
    "create_search_strategy",
]