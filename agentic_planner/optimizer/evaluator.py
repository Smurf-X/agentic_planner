# -*- coding: utf-8 -*-
"""Sample-based evaluation: cost + quality (pluggable).

This module provides:
1. PipelineEvaluator protocol - interface for all evaluators
2. StubPipelineEvaluator - deterministic stub for testing
3. LlmJudgeEvaluator - LLM-as-a-judge quality evaluation
4. RealPipelineEvaluator - runs DJ executor + collects metrics
"""

from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

from agentic_planner.contracts.cost import CostBreakdown, compute_token_cost
from agentic_planner.contracts.eval_protocol import EvalConfig, EvaluationMode
from agentic_planner.contracts.recipe import DJExecutableConfig

if TYPE_CHECKING:
    from agentic_planner.generator.llm import BaseLLMClient


@runtime_checkable
class PipelineEvaluator(Protocol):
    """Evaluate a candidate config on a sample (implementation may run DJ executor or mock)."""

    def evaluate(self, cfg: DJExecutableConfig) -> tuple[CostBreakdown, float]:
        """Return cost breakdown and mean quality in ``[0, 1]`` (or unconstrained float)."""


class StubPipelineEvaluator:
    """Deterministic stub for tests: quality from hash, zero cost."""

    def __init__(self, eval_config: EvalConfig | None = None) -> None:
        self.eval_config = eval_config or EvalConfig()

    def evaluate(self, cfg: DJExecutableConfig) -> tuple[CostBreakdown, float]:
        blob = str(cfg.get("process", [])).encode()
        q = (hash(blob) % 10000) / 10000.0
        cost = CostBreakdown(llm_token_cost=0.0, wall_time_sec=0.0)
        return cost, q


class BaseEvaluator(ABC):
    """Base class for evaluators with common utilities."""

    def __init__(self, eval_config: EvalConfig) -> None:
        self.eval_config = eval_config

    def _sample_data(
        self,
        data: List[Dict[str, Any]],
        sample_size: int | None = None,
        random_seed: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Sample data for evaluation."""
        sample_size = sample_size or self.eval_config.sample_size
        seed = random_seed or self.eval_config.random_seed

        if len(data) <= sample_size:
            return data

        rng = random.Random(seed)
        return rng.sample(data, sample_size)


class LlmJudgeEvaluator(BaseEvaluator):
    """
    LLM-as-a-judge quality evaluation.

    Evaluates pipeline outputs by having an LLM judge each sample's quality.
    """

    JUDGE_SYSTEM_PROMPT = """You are a quality evaluator for data processing outputs.
Your task is to evaluate the quality of processed data on a scale of 0-10.

Consider:
- Accuracy and completeness of the processing
- Whether the output meets the task requirements
- Quality of the transformation applied

Provide your evaluation as a JSON object:
{"score": <0-10>, "reasoning": "<brief explanation>"}"""

    JUDGE_USER_PROMPT = """## Task Description
{task_description}

## Original Input
```json
{input_data}
```

## Processed Output
```json
{output_data}
```

## Evaluation Criteria
- Score 8-10: Excellent - output perfectly meets requirements
- Score 5-7: Good - output is acceptable with minor issues
- Score 0-4: Poor - output has significant problems

Provide your evaluation as JSON."""

    def __init__(
        self,
        eval_config: EvalConfig,
        llm_client: Optional["BaseLLMClient"] = None,
        price_per_million: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Args:
            eval_config: Evaluation configuration
            llm_client: LLM client for judge calls
            price_per_million: Price per million tokens per model
        """
        super().__init__(eval_config)
        self._llm_client = llm_client
        self._price_per_million = price_per_million or {}
        self._accumulated_cost = CostBreakdown()

    def set_llm_client(self, client: "BaseLLMClient") -> None:
        """Set the LLM client for judging."""
        self._llm_client = client

    def evaluate_single(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
    ) -> tuple[float, str]:
        """
        Evaluate a single input-output pair.

        Returns:
            Tuple of (score 0-1, reasoning)
        """
        if not self._llm_client:
            raise ValueError("LLM client not set")

        user_prompt = self.JUDGE_USER_PROMPT.format(
            task_description=self.eval_config.task_description or "Data processing",
            input_data=json.dumps(input_data, ensure_ascii=False, indent=2),
            output_data=json.dumps(output_data, ensure_ascii=False, indent=2),
        )

        try:
            if hasattr(self._llm_client, "complete_json"):
                result = self._llm_client.complete_json(
                    system=self.JUDGE_SYSTEM_PROMPT,
                    user=user_prompt,
                )
            elif hasattr(self._llm_client, "generate"):
                response = self._llm_client.generate(
                    system_prompt=self.JUDGE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=0.1,
                )
                text = response.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                result = json.loads(text)
            else:
                raise ValueError("LLM client must have 'complete_json' or 'generate' method")

            score = float(result.get("score", 5)) / 10.0
            reasoning = result.get("reasoning", "")

            return score, reasoning

        except Exception as e:
            return 0.5, f"Evaluation error: {e}"

    def evaluate_batch(
        self,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
    ) -> List[tuple[float, str]]:
        """Evaluate multiple input-output pairs."""
        results = []
        for inp, out in zip(inputs, outputs):
            score, reasoning = self.evaluate_single(inp, out)
            results.append((score, reasoning))
        return results

    def evaluate_with_ground_truth(
        self,
        outputs: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        gt_key: str,
    ) -> List[tuple[float, str]]:
        """
        Compare outputs to ground truth labels.

        For simple cases, uses exact match or fuzzy matching.
        For complex cases, can still use LLM to judge similarity.
        """
        results = []
        for output, gt in zip(outputs, ground_truths):
            gt_value = gt.get(gt_key)
            output_value = output.get(gt_key)

            if gt_value is None:
                results.append((0.5, "No ground truth value"))
                continue

            if output_value is None:
                results.append((0.0, "Missing output value"))
                continue

            # Simple exact match
            if str(output_value).strip() == str(gt_value).strip():
                results.append((1.0, "Exact match"))
            else:
                # Could extend with fuzzy matching or LLM comparison
                results.append((0.0, "No match"))
        return results


@dataclass
class EvaluationResult:
    """Result of a full evaluation run."""

    cost: CostBreakdown
    quality: float
    sample_size: int
    scores: List[float] = field(default_factory=list)
    reasonings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class RealPipelineEvaluator(BaseEvaluator):
    """
    Full evaluator that runs the pipeline and evaluates outputs.

    This evaluator:
    1. Uses pre-sampled data (fixed throughout optimization)
    2. Runs the pipeline on the sample
    3. Evaluates outputs using LLM-as-a-judge or ground truth
    4. Collects cost metrics
    """

    def __init__(
        self,
        eval_config: EvalConfig,
        llm_client: Optional["BaseLLMClient"] = None,
        price_per_million: Optional[Dict[str, float]] = None,
        executor_adapter: Optional[Any] = None,
    ) -> None:
        """
        Args:
            eval_config: Evaluation configuration
            llm_client: LLM client for judge calls
            price_per_million: Token price per model
            executor_adapter: Adapter for running DJ pipelines
        """
        super().__init__(eval_config)
        self._llm_client = llm_client
        self._price_per_million = price_per_million or {}
        self._executor_adapter = executor_adapter
        self._judge_evaluator = LlmJudgeEvaluator(eval_config, llm_client, price_per_million)
        self._fixed_samples: Optional[List[Dict[str, Any]]] = None
        self._fixed_ground_truths: Optional[List[Dict[str, Any]]] = None

    def set_llm_client(self, client: "BaseLLMClient") -> None:
        """Set the LLM client."""
        self._llm_client = client
        self._judge_evaluator.set_llm_client(client)

    def set_executor_adapter(self, adapter: Any) -> None:
        """Set the executor adapter."""
        self._executor_adapter = adapter

    def prepare_fixed_samples(self, dataset_path: Optional[str] = None) -> None:
        """
        Prepare fixed samples at the start of optimization.

        This method samples the data once and stores it for reuse
        throughout the entire optimization process.

        Args:
            dataset_path: Path to the dataset (optional if already set in executor adapter)
        """
        if self._executor_adapter is None:
            return

        if self.eval_config.fixed_samples is not None:
            self._fixed_samples = self.eval_config.fixed_samples
            self._fixed_ground_truths = []
            return

        import random

        data = self._executor_adapter.load_dataset(dataset_path)

        if not data:
            return

        sample_size = min(self.eval_config.sample_size, len(data))
        rng = random.Random(self.eval_config.random_seed)
        self._fixed_samples = rng.sample(data, sample_size)

        if self.eval_config.mode == EvaluationMode.WITH_GROUND_TRUTH:
            gt_data = self._executor_adapter.load_ground_truth()
            if gt_data:
                self._fixed_ground_truths = gt_data[: len(self._fixed_samples)]
            else:
                self._fixed_ground_truths = []
        else:
            self._fixed_ground_truths = []

    def evaluate(self, cfg: DJExecutableConfig) -> tuple[CostBreakdown, float]:
        """
        Evaluate the pipeline configuration.

        Returns:
            Tuple of (cost breakdown, quality score 0-1)
        """
        result = self.evaluate_full(cfg)
        return result.cost, result.quality

    def evaluate_full(self, cfg: DJExecutableConfig) -> EvaluationResult:
        """
        Perform full evaluation with detailed results.

        Returns:
            EvaluationResult with all details
        """
        start_time = time.time()
        cost = CostBreakdown()
        scores: List[float] = []
        reasonings: List[str] = []
        errors: List[str] = []

        if self._executor_adapter is None:
            return self._stub_evaluate(cfg)

        try:
            if self._fixed_samples is None:
                self.prepare_fixed_samples()

            if self._fixed_samples is None or len(self._fixed_samples) == 0:
                return EvaluationResult(
                    cost=cost,
                    quality=0.0,
                    sample_size=0,
                    scores=[],
                    reasonings=[],
                    errors=["No samples available"],
                )

            exec_result = self._executor_adapter.run_on_fixed_samples(
                cfg,
                self._fixed_samples,
            )

            if not exec_result.ok:
                errors.extend(exec_result.errors)
                return EvaluationResult(
                    cost=cost,
                    quality=0.0,
                    sample_size=0,
                    scores=[],
                    reasonings=[],
                    errors=errors,
                )

            inputs = self._fixed_samples
            outputs = exec_result.outputs

            token_usage = exec_result.token_usage
            cost.prompt_tokens = token_usage.get("prompt_tokens", 0)
            cost.completion_tokens = token_usage.get("completion_tokens", 0)
            cost.model_usage = token_usage.get("model_usage", {})
            cost.llm_token_cost = compute_token_cost(cost.model_usage, self._price_per_million)

            if self.eval_config.mode == EvaluationMode.WITH_GROUND_TRUTH:
                gt_key = self.eval_config.ground_truth_key or ""
                ground_truths = self._fixed_ground_truths if self._fixed_ground_truths else []
                eval_results = self._judge_evaluator.evaluate_with_ground_truth(
                    outputs, ground_truths, gt_key
                )
            else:
                eval_results = self._judge_evaluator.evaluate_batch(inputs, outputs)

            for score, reasoning in eval_results:
                scores.append(score)
                reasonings.append(reasoning)

        except Exception as e:
            errors.append(str(e))

        cost.wall_time_sec = time.time() - start_time
        quality = sum(scores) / len(scores) if scores else 0.0

        return EvaluationResult(
            cost=cost,
            quality=quality,
            sample_size=len(scores),
            scores=scores,
            reasonings=reasonings,
            errors=errors,
        )

    def _stub_evaluate(self, cfg: DJExecutableConfig) -> EvaluationResult:
        """Fallback stub evaluation when no executor is available."""
        blob = str(cfg.get("process", [])).encode()
        q = (hash(blob) % 10000) / 10000.0
        return EvaluationResult(
            cost=CostBreakdown(),
            quality=q,
            sample_size=1,
            scores=[q],
            reasonings=["Stub evaluation (no executor)"],
            errors=[],
        )


# Factory function for creating evaluators


def create_evaluator(
    eval_config: EvalConfig,
    llm_client: Optional["BaseLLMClient"] = None,
    price_per_million: Optional[Dict[str, float]] = None,
    executor_adapter: Optional[Any] = None,
    use_real_executor: bool = False,
) -> PipelineEvaluator:
    """
    Factory function to create an appropriate evaluator.

    Args:
        eval_config: Evaluation configuration
        llm_client: LLM client for judge calls
        price_per_million: Token prices
        executor_adapter: DJ executor adapter
        use_real_executor: If True, use real executor; otherwise use stub

    Returns:
        A PipelineEvaluator instance
    """
    if use_real_executor and executor_adapter:
        return RealPipelineEvaluator(
            eval_config=eval_config,
            llm_client=llm_client,
            price_per_million=price_per_million,
            executor_adapter=executor_adapter,
        )

    if llm_client:
        # Can do LLM-judge evaluation but no real executor
        return RealPipelineEvaluator(
            eval_config=eval_config,
            llm_client=llm_client,
            price_per_million=price_per_million,
            executor_adapter=None,  # Will use stub for execution
        )

    # Fall back to pure stub
    return StubPipelineEvaluator(eval_config)
