# -*- coding: utf-8 -*-
"""Orchestrate directive-only, search-only, or sequential optimization.

This module provides the high-level API for running pipeline optimization:
- OptimizationRunner: Main entry point for optimization
- Supports three modes: directive_only, search_only, directive_then_search
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agentic_planner.contracts.recipe import DJExecutableConfig, save_executable_config
from agentic_planner.optimizer.directive_engine import (
    DirectiveEngine,
    DirectiveEngineConfig,
    DirectiveEngineMode,
    DirectiveEngineRun,
)
from agentic_planner.optimizer.evaluator import PipelineEvaluator, StubPipelineEvaluator
from agentic_planner.optimizer.optimization_config import OptimizationConfig
from agentic_planner.optimizer.search.beam import BeamSearchConfig, BeamSearchOptimizer, CandidateRecord

if TYPE_CHECKING:
    from agentic_planner.generator.llm import BaseLLMClient
    from agentic_planner.optimizer.executor_adapter import ExecutorAdapter


class OptimizationRunMode(str, Enum):
    """Which stages to run."""

    DIRECTIVE_ONLY = "directive_only"
    """Run only the directive engine (Stage 1)."""

    SEARCH_ONLY = "search_only"
    """Run only the search optimizer (Stage 2)."""

    DIRECTIVE_THEN_SEARCH = "directive_then_search"
    """Run Stage 1, then use result as root for Stage 2."""


@dataclass
class OptimizationRunnerResult:
    """Result of running the optimization pipeline."""

    mode: OptimizationRunMode
    ok: bool = True
    directive: Optional[DirectiveEngineRun] = None
    candidates: Optional[List[CandidateRecord]] = None
    best_config: Optional[DJExecutableConfig] = None
    best_quality: float = 0.0
    best_cost: Optional[Any] = None
    errors: List[str] = field(default_factory=list)

    def get_best_candidate(self) -> Optional[CandidateRecord]:
        """Get the best candidate from search results."""
        if not self.candidates:
            return None
        return max(self.candidates, key=lambda c: c.quality)


class OptimizationRunner:
    """
    High-level API combining stage-1 directives and stage-2 beam search.

    Usage:
        # Simple directive-only optimization
        runner = OptimizationRunner(mode=OptimizationRunMode.DIRECTIVE_ONLY)
        result = runner.run(config)

        # With full configuration
        config = OptimizationConfig.from_file("optimize.yaml")
        runner = OptimizationRunner.from_config(config)
        result = runner.run(pipeline_config)
    """

    def __init__(
        self,
        *,
        mode: OptimizationRunMode = OptimizationRunMode.DIRECTIVE_ONLY,
        directive_config: Optional[Dict[str, Any]] = None,
        beam_config: Optional[Dict[str, Any]] = None,
        evaluator: Optional[PipelineEvaluator] = None,
        llm_client: Optional["BaseLLMClient"] = None,
        executor_adapter: Optional["ExecutorAdapter"] = None,
        output_dir: Optional[str] = None,
        save_trace: bool = True,
    ) -> None:
        """
        Args:
            mode: Which optimization stages to run
            directive_config: Configuration for directive engine
            beam_config: Configuration for beam search
            evaluator: Evaluator for quality scoring
            llm_client: LLM client for inference and judging
            executor_adapter: Adapter for running pipelines
            output_dir: Directory to save optimized configs
            save_trace: Whether to save optimization trace
        """
        self.mode = mode
        self._directive_cfg = directive_config or {}
        self._beam_cfg = beam_config or {}
        self._evaluator = evaluator or StubPipelineEvaluator()
        self._llm_client = llm_client
        self._executor_adapter = executor_adapter
        self._output_dir = output_dir
        self._save_trace = save_trace

    @classmethod
    def from_config(
        cls,
        config: OptimizationConfig,
        llm_client: Optional["BaseLLMClient"] = None,
        executor_adapter: Optional["ExecutorAdapter"] = None,
    ) -> "OptimizationRunner":
        """
        Create runner from an OptimizationConfig.

        Args:
            config: Full optimization configuration
            llm_client: Optional LLM client
            executor_adapter: Optional executor adapter

        Returns:
            Configured OptimizationRunner
        """
        from agentic_planner.optimizer.evaluator import create_evaluator

        # Create evaluator
        evaluator = create_evaluator(
            eval_config=config.evaluation,
            llm_client=llm_client,
            price_per_million=config.pricing.prices,
            executor_adapter=executor_adapter,
            use_real_executor=executor_adapter is not None,
        )

        return cls(
            mode=OptimizationRunMode(config.run_mode),
            directive_config=config.directive.model_dump(),
            beam_config=config.search.model_dump() if config.search else None,
            evaluator=evaluator,
            llm_client=llm_client,
            executor_adapter=executor_adapter,
            output_dir=config.output_dir,
            save_trace=config.save_trace,
        )

    def set_llm_client(self, client: "BaseLLMClient") -> None:
        """Set the LLM client."""
        self._llm_client = client

    def set_evaluator(self, evaluator: PipelineEvaluator) -> None:
        """Set the evaluator."""
        self._evaluator = evaluator

    def run(self, cfg: DJExecutableConfig) -> OptimizationRunnerResult:
        """
        Run optimization on the given pipeline configuration.

        Args:
            cfg: The pipeline configuration to optimize

        Returns:
            OptimizationRunnerResult with optimized config(s)
        """
        current = cfg
        dir_run: Optional[DirectiveEngineRun] = None
        errors: List[str] = []

        # Stage 1: Directive-based optimization
        if self.mode in (OptimizationRunMode.DIRECTIVE_ONLY, OptimizationRunMode.DIRECTIVE_THEN_SEARCH):
            engine = DirectiveEngine(
                DirectiveEngineConfig.model_validate(self._directive_cfg),
                llm_client=self._llm_client,
            )
            dir_run = engine.run(current)
            if not dir_run.ok:
                errors.extend(dir_run.errors)
            current = dir_run.config

        # Stage 2: Search-based optimization
        if self.mode in (OptimizationRunMode.SEARCH_ONLY, OptimizationRunMode.DIRECTIVE_THEN_SEARCH):
            if not self._beam_cfg:
                errors.append("Search mode requires beam_config")
            else:
                beam = BeamSearchOptimizer(
                    BeamSearchConfig.model_validate(self._beam_cfg),
                    evaluator=self._evaluator,
                )
                candidates = beam.search(current)

                # Find best candidate
                best = max(candidates, key=lambda c: c.quality) if candidates else None

                # Save results if output_dir is set
                if self._output_dir and best:
                    self._save_results(best)

                return OptimizationRunnerResult(
                    mode=self.mode,
                    ok=len(errors) == 0,
                    directive=dir_run,
                    candidates=candidates,
                    best_config=best.config if best else current,
                    best_quality=best.quality if best else 0.0,
                    best_cost=best.cost if best else None,
                    errors=errors,
                )

        # Directive-only result
        if self._output_dir and dir_run:
            self._save_directive_result(dir_run)

        return OptimizationRunnerResult(
            mode=self.mode,
            ok=len(errors) == 0,
            directive=dir_run,
            candidates=None,
            best_config=current,
            best_quality=0.0,  # Not evaluated in directive-only mode
            best_cost=None,
            errors=errors,
        )

    def _save_results(self, best: CandidateRecord) -> None:
        """Save the best candidate to output directory."""
        if not self._output_dir:
            return

        output_path = Path(self._output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = output_path / "optimized_config.yaml"
        save_executable_config(best.config, config_path)

        # Save trace if enabled
        if self._save_trace and best.trace:
            trace_path = output_path / "optimization_trace.json"
            self._save_trace_file(best.trace, trace_path)

    def _save_directive_result(self, result: DirectiveEngineRun) -> None:
        """Save directive-only result."""
        if not self._output_dir:
            return

        output_path = Path(self._output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = output_path / "optimized_config.yaml"
        save_executable_config(result.config, config_path)

        # Save trace if enabled
        if self._save_trace and result.trace:
            trace_path = output_path / "directive_trace.json"
            self._save_trace_file(result.trace, trace_path)

    def _save_trace_file(self, trace: List[Any], path: Path) -> None:
        """Save trace to JSON file."""
        import json

        trace_data = []
        for item in trace:
            if hasattr(item, "__dict__"):
                trace_data.append({
                    "ok": getattr(item, "ok", None),
                    "applied": getattr(item, "applied", None),
                    "directive_name": getattr(item, "directive_name", None),
                    "message": getattr(item, "message", None),
                })

        with path.open("w", encoding="utf-8") as f:
            json.dump(trace_data, f, ensure_ascii=False, indent=2)


# Convenience functions


def optimize_pipeline(
    cfg: DJExecutableConfig,
    mode: str = "directive_only",
    directives: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> OptimizationRunnerResult:
    """
    Quick optimization with sensible defaults.

    Args:
        cfg: Pipeline configuration
        mode: Optimization mode
        directives: List of directives to apply
        output_dir: Where to save results

    Returns:
        OptimizationRunnerResult
    """
    runner = OptimizationRunner(
        mode=OptimizationRunMode(mode),
        directive_config={
            "mode": DirectiveEngineMode.STATIC,
            "directives": directives or ["reorder_filters_first", "remove_redundant_ops"],
        },
        output_dir=output_dir,
    )
    return runner.run(cfg)


def optimize_with_search(
    cfg: DJExecutableConfig,
    evaluator: PipelineEvaluator,
    beam_width: int = 4,
    max_iterations: int = 3,
    expansion_directives: Optional[List[str]] = None,
) -> OptimizationRunnerResult:
    """
    Run search-based optimization.

    Args:
        cfg: Pipeline configuration
        evaluator: Evaluator for quality scoring
        beam_width: Number of candidates to keep
        max_iterations: Maximum search iterations
        expansion_directives: Directives for neighbor generation

    Returns:
        OptimizationRunnerResult
    """
    runner = OptimizationRunner(
        mode=OptimizationRunMode.SEARCH_ONLY,
        evaluator=evaluator,
        beam_config={
            "beam_width": beam_width,
            "max_iterations": max_iterations,
            "expansion_directives": expansion_directives or ["tighten_filters", "loosen_filters"],
        },
    )
    return runner.run(cfg)
