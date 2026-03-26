# -*- coding: utf-8 -*-
"""Sample-based evaluation: cost + quality (pluggable).

This module provides:
1. PipelineEvaluator protocol - interface for all evaluators
2. StubPipelineEvaluator - deterministic stub for testing
3. LlmJudgeEvaluator - LLM-as-a-judge quality evaluation (DocETL-style)
4. RealPipelineEvaluator - runs DJ executor + collects metrics
"""

from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Union, runtime_checkable

from agentic_planner.contracts.cost import CostBreakdown, compute_token_cost
from agentic_planner.contracts.eval_protocol import EvalConfig, EvaluationMode
from agentic_planner.contracts.recipe import DJExecutableConfig

if TYPE_CHECKING:
    from agentic_planner.generator.llm import BaseLLMClient
    from agentic_planner.optimizer.model_registry import ModelRegistry


# Quality score mapping (DocETL style)
QUALITY_SCORES = {
    "Satisfactory": 1.0,
    "Mostly Satisfactory": 0.75,
    "Partially Satisfactory": 0.5,
    "Unsatisfactory": 0.25,
}


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
    LLM-as-a-judge quality evaluation (DocETL-style).

    Evaluates pipeline outputs by:
    1. Generating a validator prompt based on pipeline config
    2. Having an LLM judge each sample's quality
    3. Using 4-level quality scoring (Satisfactory, Mostly, Partially, Unsatisfactory)
    """

    VALIDATOR_GENERATION_PROMPT = """You are an AI assistant tasked with creating custom validation prompts for data processing operations.

Analyze the following pipeline and its input/output:

Pipeline Operators: {operators}
Operator Details: {operator_details}
Sample Input: {sample_input}
Sample Output: {sample_output}

Based on this information, create a custom validator prompt (2-3 sentences) that will assess how well the pipeline performed its intended task.

The validator prompt should ask specific questions about:
1. Recall: Did the pipeline correctly identify/process all relevant data?
2. Precision: Is the output accurate without errors?
3. Quality: Are there any issues or inconsistencies?

Important: Tailor the prompt to the specific operators. For example:
- If filtering data: ask if low-quality data was correctly removed while keeping useful content
- If transforming data: ask if the transformation is accurate and complete
- If using LLM: ask if the LLM output is correct and free of hallucinations

Return a JSON object: {{"validator_prompt": "<your prompt>"}}"""

    QUALITY_SYSTEM_PROMPT = """You are an AI assistant tasked with evaluating the quality of data processing outputs.

Assess the output based on the validator prompt criteria. Be strict but fair."""

    QUALITY_USER_PROMPT = """## Validator Criteria
{validator_prompt}

## Original Input
```json
{input_data}
```

## Processed Output
```json
{output_data}
```

Based on the validator criteria, categorize the quality into one of:
- "Satisfactory": The output fully meets all criteria
- "Mostly Satisfactory": The output meets most criteria with minor issues
- "Partially Satisfactory": The output meets some criteria but has problems
- "Unsatisfactory": The output fails to meet the criteria

Provide your response as JSON: {{"quality": "<category>", "reason": "<brief explanation>"}}"""

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
        self._cached_validator_prompt: Optional[str] = None

    def set_llm_client(self, client: "BaseLLMClient") -> None:
        """Set the LLM client for judging."""
        self._llm_client = client

    def _generate_validator_prompt(
        self,
        pipeline_config: DJExecutableConfig,
        sample_input: Dict[str, Any],
        sample_output: Dict[str, Any],
    ) -> str:
        """
        Generate a validator prompt based on pipeline config and sample data.

        This method creates a task-specific evaluation prompt by:
        1. Extracting operator details from pipeline config
        2. Using LLM to generate a tailored validator prompt

        Args:
            pipeline_config: The pipeline configuration
            sample_input: A sample input record
            sample_output: The corresponding output record

        Returns:
            A validator prompt string
        """
        if self._cached_validator_prompt:
            return self._cached_validator_prompt

        if self.eval_config.task_description:
            self._cached_validator_prompt = self.eval_config.task_description
            return self._cached_validator_prompt

        operator_details = self._extract_operator_details(pipeline_config)
        operators = self._extract_operator_names(pipeline_config)

        if not self._llm_client:
            return self._get_default_validator_prompt(operator_details)

        try:
            prompt = self.VALIDATOR_GENERATION_PROMPT.format(
                operators=operators,
                operator_details=operator_details,
                sample_input=json.dumps(sample_input, ensure_ascii=False, indent=2)[:500],
                sample_output=json.dumps(sample_output, ensure_ascii=False, indent=2)[:500],
            )

            if hasattr(self._llm_client, "generate"):
                response = self._llm_client.generate(
                    system_prompt="You are a helpful assistant that creates validation prompts for data pipelines.",
                    user_prompt=prompt,
                    temperature=0.3,
                )
                text = response.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                result = json.loads(text)
                validator_prompt = result.get("validator_prompt", "")
            elif hasattr(self._llm_client, "complete_json"):
                result = self._llm_client.complete_json(
                    system="You are a helpful assistant that creates validation prompts for data pipelines.",
                    user=prompt,
                )
                validator_prompt = result.get("validator_prompt", "")
            else:
                validator_prompt = self._get_default_validator_prompt(operator_details)

            if validator_prompt:
                self._cached_validator_prompt = validator_prompt
                return validator_prompt

            return self._get_default_validator_prompt(operator_details)

        except Exception as e:
            return self._get_default_validator_prompt(operator_details)

    def _get_default_validator_prompt(self, operator_details: str) -> str:
        """Get a default validator prompt based on operator details."""
        details_lower = operator_details.lower()

        if "filter" in details_lower:
            if "language" in details_lower:
                return "Did the pipeline correctly identify and keep only the target language content while removing non-target language data?"
            elif "length" in details_lower:
                return "Did the pipeline correctly filter out texts that are too short or too long while keeping appropriately sized content?"
            else:
                return "Did the pipeline correctly filter out low-quality or irrelevant data while retaining useful content?"
        elif "mapper" in details_lower:
            return "Did the pipeline correctly transform the input data? Is the output accurate and complete?"
        else:
            return "Did the pipeline process the data correctly? Is the output accurate and complete?"

    def _extract_operator_names(self, pipeline_config: DJExecutableConfig) -> str:
        """Extract operator names from pipeline config."""
        process = pipeline_config.get("process", [])
        if not process:
            return "unknown"

        op_names = []
        for step in process:
            if isinstance(step, dict):
                op_names.extend(list(step.keys()))

        return ", ".join(op_names) if op_names else "unknown"

    def _extract_operator_details(self, pipeline_config: DJExecutableConfig) -> str:
        """
        Extract detailed operator information from pipeline config.

        Returns a human-readable description of what each operator does.
        """
        process = pipeline_config.get("process", [])
        if not process:
            return "unknown"

        details = []
        for step in process:
            if isinstance(step, dict):
                for op_name, op_params in step.items():
                    detail = self._describe_operator(op_name, op_params or {})
                    details.append(detail)

        return "; ".join(details) if details else "unknown"

    def _describe_operator(self, op_name: str, params: Dict[str, Any]) -> str:
        """Generate a human-readable description of an operator."""
        if "text_length_filter" in op_name:
            min_len = params.get("min_len", 0)
            max_len = params.get("max_len", float("inf"))
            return f"Filter text by length: keep texts with {min_len}-{max_len} characters"

        elif "language_id_score_filter" in op_name:
            lang = params.get("lang", "unknown")
            min_score = params.get("min_score", 0)
            return f"Filter by language: keep {lang} text with confidence >= {min_score}"

        elif "words_num_filter" in op_name:
            min_num = params.get("min_num", 0)
            max_num = params.get("max_num", float("inf"))
            return f"Filter by word count: keep texts with {min_num}-{max_num} words"

        elif "special_characters_filter" in op_name:
            max_ratio = params.get("max_ratio", 1.0)
            return f"Filter by special character ratio: keep texts with special chars <= {max_ratio}"

        elif "perplexity_filter" in op_name:
            max_ppl = params.get("max_ppl", float("inf"))
            return f"Filter by perplexity: keep texts with perplexity <= {max_ppl}"

        elif "llm" in op_name.lower():
            model = params.get("api_or_hf_model", params.get("api_model", "unknown"))
            return f"LLM operator using model {model}"

        elif "mapper" in op_name:
            return f"Transform data using {op_name}"

        elif "filter" in op_name:
            return f"Filter data using {op_name}"

        else:
            return f"Process data using {op_name}"

    def evaluate_single(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        validator_prompt: Optional[str] = None,
    ) -> tuple[float, str]:
        """
        Evaluate a single input-output pair.

        Args:
            input_data: Input record
            output_data: Output record
            validator_prompt: Optional custom validator prompt

        Returns:
            Tuple of (score 0-1, reasoning)
        """
        if not self._llm_client:
            return 0.5, "No LLM client available"

        vp = validator_prompt or "Is the output correct and complete?"

        user_prompt = self.QUALITY_USER_PROMPT.format(
            validator_prompt=vp,
            input_data=json.dumps(input_data, ensure_ascii=False, indent=2),
            output_data=json.dumps(output_data, ensure_ascii=False, indent=2),
        )

        try:
            if hasattr(self._llm_client, "complete_json"):
                result = self._llm_client.complete_json(
                    system=self.QUALITY_SYSTEM_PROMPT,
                    user=user_prompt,
                )
            elif hasattr(self._llm_client, "generate"):
                response = self._llm_client.generate(
                    system_prompt=self.QUALITY_SYSTEM_PROMPT,
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

            quality = result.get("quality", "Partially Satisfactory")
            reasoning = result.get("reason", "")

            score = QUALITY_SCORES.get(quality, 0.5)

            return score, reasoning

        except Exception as e:
            return 0.5, f"Evaluation error: {e}"

    def evaluate_batch(
        self,
        inputs: List[Dict[str, Any]],
        outputs: List[Dict[str, Any]],
        pipeline_config: Optional[DJExecutableConfig] = None,
    ) -> List[tuple[float, str]]:
        """
        Evaluate multiple input-output pairs.

        Args:
            inputs: List of input records
            outputs: List of output records
            pipeline_config: Optional pipeline config for generating validator prompt

        Returns:
            List of (score, reasoning) tuples
        """
        validator_prompt = None

        if pipeline_config and inputs and outputs:
            validator_prompt = self._generate_validator_prompt(
                pipeline_config, inputs[0], outputs[0]
            )

        results = []
        for inp, out in zip(inputs, outputs):
            score, reasoning = self.evaluate_single(inp, out, validator_prompt)
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
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    inputs: List[Dict[str, Any]] = field(default_factory=list)


class RealPipelineEvaluator(BaseEvaluator):
    """
    Full evaluator that runs the pipeline and evaluates outputs.

    This evaluator:
    1. Uses pre-sampled data (fixed throughout optimization)
    2. Runs the pipeline on the sample
    3. Evaluates outputs using LLM-as-a-judge or ground truth
    4. Collects cost metrics using ModelRegistry prices
    """

    def __init__(
        self,
        eval_config: EvalConfig,
        llm_client: Optional["BaseLLMClient"] = None,
        price_per_million: Optional[Dict[str, float]] = None,
        executor_adapter: Optional[Any] = None,
        model_registry: Optional["ModelRegistry"] = None,
    ) -> None:
        """
        Args:
            eval_config: Evaluation configuration
            llm_client: LLM client for judge calls
            price_per_million: Token price per model (deprecated, use model_registry)
            executor_adapter: Adapter for running DJ pipelines
            model_registry: ModelRegistry for price lookup and client creation
        """
        super().__init__(eval_config)
        self._llm_client = llm_client
        self._executor_adapter = executor_adapter
        self._model_registry = model_registry

        if model_registry:
            self._price_per_million = model_registry.get_price_table()
        else:
            self._price_per_million = price_per_million or {}

        self._judge_evaluator = LlmJudgeEvaluator(
            eval_config,
            llm_client,
            self._price_per_million,
        )
        self._fixed_samples: Optional[List[Dict[str, Any]]] = None
        self._fixed_ground_truths: Optional[List[Dict[str, Any]]] = None

    def set_llm_client(self, client: "BaseLLMClient") -> None:
        """Set the LLM client."""
        self._llm_client = client
        self._judge_evaluator.set_llm_client(client)

    def set_executor_adapter(self, adapter: Any) -> None:
        """Set the executor adapter."""
        self._executor_adapter = adapter

    def set_model_registry(self, registry: "ModelRegistry") -> None:
        """Set the model registry for price lookup."""
        self._model_registry = registry
        self._price_per_million = registry.get_price_table()
        self._judge_evaluator._price_per_million = self._price_per_million

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
                    outputs=[],
                    inputs=[],
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
                    outputs=[],
                    inputs=[],
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
                eval_results = self._judge_evaluator.evaluate_batch(
                    inputs, outputs, pipeline_config=cfg
                )

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
            outputs=outputs if outputs else [],
            inputs=inputs if inputs else [],
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
            outputs=[],
            inputs=[],
        )


# Factory function for creating evaluators


def create_evaluator(
    eval_config: EvalConfig,
    llm_client: Optional["BaseLLMClient"] = None,
    price_per_million: Optional[Dict[str, float]] = None,
    executor_adapter: Optional[Any] = None,
    use_real_executor: bool = False,
    model_registry: Optional["ModelRegistry"] = None,
) -> PipelineEvaluator:
    """
    Factory function to create an appropriate evaluator.

    Args:
        eval_config: Evaluation configuration
        llm_client: LLM client for judge calls
        price_per_million: Token prices (deprecated, use model_registry)
        executor_adapter: DJ executor adapter
        use_real_executor: If True, use real executor; otherwise use stub
        model_registry: ModelRegistry for price lookup

    Returns:
        A PipelineEvaluator instance
    """
    if use_real_executor and executor_adapter:
        return RealPipelineEvaluator(
            eval_config=eval_config,
            llm_client=llm_client,
            price_per_million=price_per_million,
            executor_adapter=executor_adapter,
            model_registry=model_registry,
        )

    if llm_client or model_registry:
        return RealPipelineEvaluator(
            eval_config=eval_config,
            llm_client=llm_client,
            price_per_million=price_per_million,
            executor_adapter=None,
            model_registry=model_registry,
        )

    return StubPipelineEvaluator(eval_config)
