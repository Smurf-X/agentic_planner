# -*- coding: utf-8 -*-
"""
End-to-end optimizer example with ModelRegistry support.

This script demonstrates how to run the pipeline optimizer with:
- 4 operators (2 filters + 2 LLM operators)
- Fixed sampling (configurable samples)
- BeamSearch strategy with LLM-guided action selection
- Multiple model support via models.yaml

Usage:
    # Stub mode (no API calls, for testing)
    python examples/run_optimizer.py --stub --sample-size 10

    # With model configuration file
    python examples/run_optimizer.py --models-config examples/models.yaml

    # Or use environment variables
    export OPENAI_API_KEY="your-key"
    python examples/run_optimizer.py --sample-size 10
"""

import argparse
import json
import os
import re
from pathlib import Path

from agentic_planner import (
    DJExecutableConfig,
    EvalConfig,
    BeamSearchConfig,
    BeamSearchStrategy,
    save_executable_config,
)
from agentic_planner.optimizer.evaluator import RealPipelineEvaluator, StubPipelineEvaluator
from agentic_planner.optimizer.executor_adapter import DJExecutorAdapter, StubExecutorAdapter
from agentic_planner.optimizer.action import ActionSpaceBuilder
from agentic_planner.optimizer.llm_action_selector import LLMActionSelector
from agentic_planner.optimizer.directives import (
    TightenFiltersDirective,
    LoosenFiltersDirective,
)
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
    dataset_path: str, output_path: str, llm_model: str = "gpt-4o-mini"
) -> DJExecutableConfig:
    """Create initial pipeline configuration with 4 operators."""
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
                    "max_len": 500,
                }
            },
            {
                "language_id_score_filter": {
                    "lang": "en",
                    "min_score": 0.8,
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
    parser.add_argument(
        "--selector-model",
        default="",
        help="Model for action selection (defaults to judge model)",
    )

    # Search configuration
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of samples for evaluation",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=3,
        help="Beam width",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum iterations",
    )
    parser.add_argument(
        "--llm-selection-top-k",
        type=int,
        default=10,
        help="Number of actions for LLM to select",
    )
    parser.add_argument(
        "--no-llm-selection",
        action="store_true",
        help="Disable LLM-guided action selection",
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

    # Create pipeline config
    print("\n[2/7] Creating initial pipeline config...")
    initial_config = create_pipeline_config(
        dataset_path, output_path, llm_model=args.pipeline_llm_model
    )
    print(f"  - Dataset: {dataset_path}")
    print(f"  - Output: {output_path}")
    print(f"  - Operators: {len(initial_config['process'])}")
    print(f"  - Pipeline LLM model: {args.pipeline_llm_model}")

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

    # Configure action space
    print("\n[5/7] Configuring action space...")

    action_builder = ActionSpaceBuilder(
        directives=[
            TightenFiltersDirective(intensity=0.1),
            LoosenFiltersDirective(intensity=0.1),
        ],
        model_registry=registry if not args.stub else None,
    )
    print(f"  - Directives: tighten_filters, loosen_filters")
    if not args.stub and registry:
        print(f"  - Model swap: {len(registry.get_candidate_models())} candidate models")

    # Setup LLM Action Selector
    print("\n[6/7] Setting up LLM Action Selector...")

    use_llm_selection = not args.no_llm_selection

    if use_llm_selection and not args.stub:
        selector_model = args.selector_model
        if not selector_model:
            candidate_models = registry.get_candidate_models()
            if candidate_models:
                selector_model = candidate_models[0]
            else:
                selector_model = registry.list_models()[0] if registry.list_models() else None

        if selector_model:
            selector_client = registry.create_client(selector_model)
            llm_selector = LLMActionSelector(
                llm_client=selector_client,
                model=selector_model,
            )
            print(f"  - Enabled: True")
            print(f"  - Selector model: {selector_model}")
            print(f"  - Top-k: {args.llm_selection_top_k}")
        else:
            llm_selector = None
            print(f"  - Enabled: False (no model available)")
    else:
        llm_selector = None
        print(f"  - Enabled: False (fallback to first-k)")

    # Configure BeamSearch
    print("\n[7/7] Configuring BeamSearch strategy...")

    beam_config = BeamSearchConfig(
        beam_width=args.beam_width,
        max_iterations=args.max_iterations,
        use_llm_selection=use_llm_selection,
        llm_selection_top_k=args.llm_selection_top_k,
        track_pareto=True,
        cost_weight=0.3,
    )

    strategy = BeamSearchStrategy(
        config=beam_config,
        evaluator=evaluator,
        action_builder=action_builder,
        llm_selector=llm_selector,
    )
    strategy.set_data_sample(data_sample)

    print(f"  - Beam width: {args.beam_width}")
    print(f"  - Max iterations: {args.max_iterations}")
    print(f"  - LLM selection: {use_llm_selection}")

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
