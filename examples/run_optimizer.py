# -*- coding: utf-8 -*-
"""
End-to-end optimizer example.

This script demonstrates how to run the pipeline optimizer with:
- 4 operators (2 filters + 2 LLM operators)
- Fixed sampling (10 samples)
- BeamSearch strategy

Usage:
    # Stub mode (no API calls, for testing)
    python examples/run_optimizer.py --stub --sample-size 10

    # Real mode with API
    python examples/run_optimizer.py \
        --judge-api-key YOUR_API_KEY \
        --judge-base-url https://api.openai.com/v1 \
        --judge-model gpt-4o-mini
"""

import argparse
from pathlib import Path

from agentic_planner import (
    DJExecutableConfig,
    EvalConfig,
    BeamSearchConfig,
    BeamSearchStrategy,
    save_executable_config,
)
from agentic_planner.generator.http_llm import OpenAICompatibleJsonClient
from agentic_planner.optimizer.evaluator import RealPipelineEvaluator, StubPipelineEvaluator
from agentic_planner.optimizer.executor_adapter import StubExecutorAdapter
from agentic_planner.optimizer.action import ActionSpaceBuilder
from agentic_planner.optimizer.directives import (
    TightenFiltersDirective,
    LoosenFiltersDirective,
)


def create_pipeline_config(
    dataset_path: str, output_path: str, llm_model: str = "gpt-4o-mini"
) -> DJExecutableConfig:
    """
    Create initial pipeline configuration with 4 operators.

    Args:
        dataset_path: Path to input dataset
        output_path: Path for output
        llm_model: Model to use for LLM operators in pipeline
    """
    return {
        "dataset_path": dataset_path,
        "export_path": output_path,
        "process": [
            {
                "language_id_score_filter": {
                    "lang": "en",
                    "min_score": 0.8,
                }
            },
            {
                "text_length_filter": {
                    "min_len": 20,
                    "max_len": 500,
                }
            },
            {
                "llm_quality_score_filter": {
                    "api_model": llm_model,
                    "min_score": 0.5,
                }
            },
            {
                "extract_keyword_mapper": {
                    "api_model": llm_model,
                }
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run pipeline optimizer")

    # Judge LLM configuration (for LLM-as-Judge evaluation)
    parser.add_argument("--judge-api-key", default="", help="LLM API key for Judge")
    parser.add_argument(
        "--judge-base-url", default="https://api.openai.com/v1", help="LLM API base URL for Judge"
    )
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Model for LLM-as-Judge")

    # Pipeline LLM configuration (for operators inside pipeline)
    parser.add_argument(
        "--pipeline-llm-model", default="gpt-4o-mini", help="Model for Pipeline LLM operators"
    )

    # Search configuration
    parser.add_argument(
        "--sample-size", type=int, default=10, help="Number of samples for evaluation"
    )
    parser.add_argument("--beam-width", type=int, default=3, help="Beam width")
    parser.add_argument("--max-iterations", type=int, default=2, help="Maximum iterations")
    parser.add_argument(
        "--stub", action="store_true", help="Use stub evaluator (no real API calls)"
    )
    args = parser.parse_args()

    project_dir = Path(__file__).parent.parent
    dataset_path = str(project_dir / "data" / "qa_data.jsonl")
    output_path = str(project_dir / "output" / "optimized.jsonl")

    (project_dir / "output").mkdir(exist_ok=True)

    print("=" * 60)
    print("Pipeline Optimizer - End-to-End Example")
    print("=" * 60)

    print("\n[1/5] Creating initial pipeline config...")
    initial_config = create_pipeline_config(
        dataset_path, output_path, llm_model=args.pipeline_llm_model
    )
    print(f"  - Dataset: {dataset_path}")
    print(f"  - Output: {output_path}")
    print(f"  - Operators: {len(initial_config['process'])}")
    print(f"  - Pipeline LLM model: {args.pipeline_llm_model}")

    print("\n[2/5] Setting up evaluator...")
    eval_config = EvalConfig(
        sample_size=args.sample_size,
        random_seed=42,
        task_description="Clean and improve English text data",
    )

    if args.stub or not args.judge_api_key:
        print("  - Using STUB evaluator (no real API calls)")
        print("  - Judge model: N/A (stub mode)")
        evaluator = StubPipelineEvaluator(eval_config)
    else:
        print(f"  - Judge model: {args.judge_model}")
        print(f"  - Judge base URL: {args.judge_base_url}")

        judge_client = OpenAICompatibleJsonClient(
            model=args.judge_model,
            api_key=args.judge_api_key,
            base_url=args.judge_base_url,
        )

        executor_adapter = StubExecutorAdapter(dataset_path=dataset_path)

        evaluator = RealPipelineEvaluator(
            eval_config=eval_config,
            llm_client=judge_client,
            executor_adapter=executor_adapter,
        )
    print(f"  - Sample size: {args.sample_size}")

    print("\n[3/5] Configuring action space...")
    action_builder = ActionSpaceBuilder(
        directives=[
            TightenFiltersDirective(intensity=0.1),
            LoosenFiltersDirective(intensity=0.1),
        ]
    )
    print(f"  - Directives: tighten_filters, loosen_filters")

    print("\n[4/5] Configuring BeamSearch strategy...")
    beam_config = BeamSearchConfig(
        beam_width=args.beam_width,
        max_iterations=args.max_iterations,
        track_pareto=True,
        cost_weight=0.3,
    )

    strategy = BeamSearchStrategy(
        config=beam_config,
        evaluator=evaluator,
        action_builder=action_builder,
    )
    print(f"  - Beam width: {args.beam_width}")
    print(f"  - Max iterations: {args.max_iterations}")

    print("\n[5/5] Running optimization...")
    print("-" * 60)

    report = strategy.search(initial_config)

    print("-" * 60)
    print("\n[Results]")
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

    print("\n[Saved files]")
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
