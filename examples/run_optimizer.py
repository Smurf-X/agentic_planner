# -*- coding: utf-8 -*-
"""
End-to-end optimizer example with ModelRegistry support.

This script demonstrates how to run the pipeline optimizer with:
- 4 operators (2 basic filters + 2 LLM operators)
- Fixed sampling (configurable samples)
- MOAR strategy (root-only baseline for Task 1)
- Multiple model support via models.yaml

Pipeline stages:
1. text_length_filter - Filter by text length (no LLM, fast)
2. language_id_score_filter - Filter by language (no LLM, fast)
3. words_num_filter - Filter by word count (no LLM)
4. special_characters_filter - Filter by special character ratio (no LLM)

Usage:
    # Stub mode (no API calls, for testing)
    python examples/run_optimizer.py --stub --sample-size 10

    # With model configuration file
    python examples/run_optimizer.py --models-config examples/models.yaml

    # Or use environment variables
    export OPENAI_API_KEY="your-key"
    python examples/run_optimizer.py --sample-size 10
"""

# Suppress Data-Juicer logging before any imports
import os
os.environ["TQDM_DISABLE"] = "1"

import logging
logging.getLogger("data_juicer").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.ERROR)

# Suppress loguru output from Data-Juicer
try:
    from loguru import logger
    logger.remove()
    logger.add(lambda msg: None, level="WARNING")
except ImportError:
    pass

import argparse
import json
import re
from pathlib import Path

from agentic_planner import (
    DJExecutableConfig,
    EvalConfig,
    MOARSearchConfig,
    MOARSearchStrategy,
    save_executable_config,
)
from agentic_planner.optimizer.evaluator import RealPipelineEvaluator, StubPipelineEvaluator
from agentic_planner.optimizer.executor_adapter import DJExecutorAdapter, StubExecutorAdapter
from agentic_planner.optimizer.model_registry import ModelRegistry, ModelsConfig


def expand_env_vars(value):
    """Recursively expand environment variables in strings."""
    if isinstance(value, str):
        pattern = r"\$\{([^}]+)\}"

        def replace(match):
            var_name = match.group(1)
            return os.environ.get(var_name, "")

        return re.sub(pattern, replace, value)
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value


def create_pipeline_config(
    dataset_path: str,
    output_path: str,
    llm_model: str = "gpt-4o-mini",
    api_endpoint: str = "",
    api_key: str = "",
) -> DJExecutableConfig:
    """Create initial pipeline configuration with 4 non-LLM operators.
    
    Pipeline stages:
    1. text_length_filter - Filter by text length (no LLM)
    2. language_id_score_filter - Filter by language (no LLM)
    3. words_num_filter - Filter by word count (no LLM)
    4. special_characters_filter - Filter by special character ratio (no LLM)
    
    All operators are non-LLM based, so no API calls are needed during execution.
    The optimizer can adjust parameters like min_len, max_len, min_score, etc.
    """
    import os

    abs_dataset_path = os.path.abspath(dataset_path)
    abs_output_path = os.path.abspath(output_path)

    return {
        "dataset_path": abs_dataset_path,
        "export_path": abs_output_path,
        "process": [
            {
                "text_length_filter": {
                    "min_len": 20,
                    "max_len": 1000,
                }
            },
            {
                "language_id_score_filter": {
                    "lang": "en",
                    "min_score": 0.8,
                }
            },
            {
                "words_num_filter": {
                    "min_num": 5,
                    "max_num": 200,
                }
            },
            {
                "special_characters_filter": {
                    "max_ratio": 0.3,
                }
            },
        ],
    }


def load_data_sample(dataset_path: str, sample_size: int) -> list:
    """Load data sample from dataset."""
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Run pipeline optimizer with ModelRegistry")

    # Configuration file
    parser.add_argument(
        "--models-config",
        default="",
        help="Path to models.yaml configuration file",
    )

    # Override options (when not using config file)
    parser.add_argument(
        "--judge-api-key",
        default="",
        help="LLM API key for Judge (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--judge-base-url",
        default="https://api.openai.com/v1",
        help="LLM API base URL for Judge",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini",
        help="Model for LLM-as-Judge",
    )
    parser.add_argument(
        "--pipeline-llm-model",
        default="gpt-4o-mini",
        help="Initial model for Pipeline LLM operators",
    )

    # Search configuration
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of samples for evaluation",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum iterations",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use stub evaluator (no real API calls)",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    dataset_path = str(project_dir / "data" / "qa_data.jsonl")
    output_path = str(project_dir / "output" / "optimized.jsonl")

    (project_dir / "output").mkdir(exist_ok=True)

    print("=" * 60)
    print("Pipeline Optimizer - End-to-End Example")
    print("=" * 60)

    # Load model registry
    print("\n[1/7] Loading model configuration...")

    if args.models_config:
        config_path = args.models_config
        if not Path(config_path).is_absolute():
            config_path = str(project_dir / config_path)

        print(f"  - Config file: {config_path}")

        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        expanded_config = expand_env_vars(raw_config)
        models_config = ModelsConfig.model_validate(expanded_config)
        registry = ModelRegistry(models_config)

        print(f"  - Judge model: {registry.get_judge_config().model}")
        print(f"  - Pipeline models: {registry.list_models()}")
        print(f"  - Candidate models: {registry.get_candidate_models()}")
    else:
        print("  - Using environment variables")
        registry = ModelRegistry.default()

        if args.judge_api_key:
            registry._config.judge.api_key = args.judge_api_key
        if args.judge_base_url:
            registry._config.judge.base_url = args.judge_base_url
        if args.judge_model:
            registry._config.judge.model = args.judge_model

        print(f"  - Judge model: {registry.get_judge_config().model}")

    # Determine default LLM model for pipeline
    if args.pipeline_llm_model == "gpt-4o-mini" and registry:
        candidate_models = registry.get_candidate_models()
        if candidate_models:
            default_llm_model = candidate_models[0]
        else:
            models = registry.list_models()
            default_llm_model = models[0] if models else args.pipeline_llm_model
    else:
        default_llm_model = args.pipeline_llm_model

    # Get API endpoint and key from registry
    api_endpoint = ""
    api_key = ""
    if registry:
        model_config = registry.get_model(default_llm_model)
        if model_config:
            api_endpoint = model_config.base_url or ""
            api_key = model_config.api_key or ""
            # Set environment variable for Data-Juicer LLM operators
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key

    # Create pipeline config
    print("\n[2/7] Creating initial pipeline config...")
    initial_config = create_pipeline_config(
        dataset_path,
        output_path,
        llm_model=default_llm_model,
        api_endpoint=api_endpoint,
        api_key=api_key,
    )
    print(f"  - Dataset: {dataset_path}")
    print(f"  - Output: {output_path}")
    print(f"  - Operators: {len(initial_config['process'])}")
    print(f"  - Pipeline LLM model: {default_llm_model}")
    if api_endpoint:
        print(f"  - API endpoint: {api_endpoint}")

    # Load data sample
    print("\n[3/7] Loading data sample...")
    data_sample = load_data_sample(dataset_path, args.sample_size)
    print(f"  - Sample size: {len(data_sample)}")

    # Setup evaluator
    print("\n[4/7] Setting up evaluator...")

    eval_config = EvalConfig(
        sample_size=args.sample_size,
        random_seed=42,
        task_description="Clean and improve English text data",
    )

    if args.stub:
        print("  - Using STUB evaluator (no real API calls)")
        evaluator = StubPipelineEvaluator(eval_config)
    else:
        judge_client = registry.create_judge_client()
        # Note: DJExecutorAdapter has a path issue on Windows due to DJ's logger
        # using relative paths with ".." which Windows doesn't allow in filenames.
        # Using StubExecutorAdapter for now until DJ fixes this.
        # To use real execution, run on Linux/macOS or wait for DJ fix.
        import platform

        if platform.system() == "Windows":
            print("  - Using STUB executor (DJ has Windows path issues)")
            executor_adapter = StubExecutorAdapter(dataset_path=dataset_path)
        else:
            executor_adapter = DJExecutorAdapter(
                dataset_path=dataset_path,
                work_dir=str(project_dir / ".dj_work"),
            )

        evaluator = RealPipelineEvaluator(
            eval_config=eval_config,
            llm_client=judge_client,
            executor_adapter=executor_adapter,
            model_registry=registry,
        )
        print(f"  - Judge model: {registry.get_judge_config().model}")
        print(f"  - Price table: {len(registry.get_price_table())} models")

    print(f"  - Sample size: {args.sample_size}")

    # Configure search
    print("\n[5/7] Configuring action space...")
    print("  - MOAR baseline: evaluate root configuration only")

    # LLM action selection is not used in the Task 1 MOAR baseline.
    print("\n[6/7] Setting up LLM Action Selector...")
    print("  - Enabled: False")

    # Configure MOAR search
    print("\n[7/7] Configuring MOAR strategy...")

    moar_config = MOARSearchConfig(
        max_iterations=args.max_iterations,
    )

    strategy = MOARSearchStrategy(
        config=moar_config,
        evaluator=evaluator,
    )

    print(f"  - Max iterations: {args.max_iterations}")
    print(f"  - Strategy alias: mcts/moar")

    # Run optimization
    print("\n" + "=" * 60)
    print("Running optimization...")
    print("=" * 60)

    report = strategy.search(initial_config)

    print("\n" + "-" * 60)
    print("[Results]")
    print("-" * 60)
    print(f"  - Success: {report.ok}")
    print(f"  - Total candidates: {len(report.candidates)}")
    print(f"  - Total evaluations: {report.total_evaluations}")
    print(f"  - Pareto front size: {len(report.pareto_front)}")

    if report.best_by_quality:
        print(f"\n  Best by quality:")
        print(f"    - Quality: {report.best_by_quality.quality:.4f}")
        print(f"    - Cost: ${report.best_by_quality.cost.llm_token_cost:.4f}")
        print(f"    - Origin: {report.best_by_quality.origin}")

    if report.best_by_cost:
        print(f"\n  Best by cost:")
        print(f"    - Quality: {report.best_by_cost.quality:.4f}")
        print(f"    - Cost: ${report.best_by_cost.cost.llm_token_cost:.4f}")
        print(f"    - Origin: {report.best_by_cost.origin}")

    if report.best_balanced:
        print(f"\n  Best balanced:")
        print(f"    - Quality: {report.best_balanced.quality:.4f}")
        print(f"    - Cost: ${report.best_balanced.cost.llm_token_cost:.4f}")
        print(f"    - Origin: {report.best_balanced.origin}")

    if report.pareto_front:
        print(f"\n  Pareto front ({len(report.pareto_front)} candidates):")
        for i, candidate in enumerate(report.pareto_front):
            print(
                f"    [{i + 1}] Q={candidate.quality:.4f}, C=${candidate.cost.llm_token_cost:.4f}"
            )
            save_executable_config(
                candidate.config,
                str(project_dir / "output" / f"pareto_{i + 1}.yaml"),
            )

    print(f"\n[Saved files]")
    print(f"  - pareto_*.yaml in {project_dir / 'output'}")

    if report.errors:
        print(f"\n[Errors] {len(report.errors)}:")
        for error in report.errors[:5]:
            print(f"  - {error}")

    print("\n" + "=" * 60)
    print("Optimization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
