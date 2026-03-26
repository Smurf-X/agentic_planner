# -*- coding: utf-8 -*-
"""
Executor adapter for running Data-Juicer pipelines on sample data.

This module provides the bridge between the optimizer and the actual
Data-Juicer pipeline execution, including:
1. Sample data loading and sampling
2. Pipeline execution on samples
3. Token usage collection
4. Output collection for evaluation
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from agentic_planner.contracts.recipe import DJExecutableConfig


@dataclass
class SampleExecutionResult:
    """Result of running a pipeline on a sample."""

    ok: bool
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: Dict[str, Any] = field(default_factory=dict)
    ground_truths: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    sample_size: int = 0
    wall_time_sec: float = 0.0


class ExecutorAdapter:
    """
    Base class for pipeline execution adapters.

    Provides a unified interface for running DJ pipelines on sample data,
    collecting outputs, and tracking token usage.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            dataset_path: Path to the input dataset (JSONL format)
            ground_truth_path: Optional path to ground truth labels
        """
        self._dataset_path = dataset_path
        self._ground_truth_path = ground_truth_path
        self._dataset_cache: Optional[List[Dict[str, Any]]] = None
        self._ground_truth_cache: Optional[List[Dict[str, Any]]] = None

    def load_dataset(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load the input dataset from a JSONL file."""
        p = path or self._dataset_path
        if not p:
            return []

        data = []
        path_obj = Path(p)
        if not path_obj.exists():
            return []

        with path_obj.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if path == self._dataset_path:
            self._dataset_cache = data

        return data

    def load_ground_truth(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load ground truth labels from a JSONL file."""
        p = path or self._ground_truth_path
        if not p:
            return []

        data = []
        path_obj = Path(p)
        if not path_obj.exists():
            return []

        with path_obj.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if path == self._ground_truth_path:
            self._ground_truth_cache = data

        return data

    def sample_data(
        self,
        data: List[Dict[str, Any]],
        sample_size: int,
        random_seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Sample a subset of the data."""
        if len(data) <= sample_size:
            return data.copy()
        rng = random.Random(random_seed)
        return rng.sample(data, sample_size)

    def run_sample(
        self,
        cfg: DJExecutableConfig,
        sample_size: int = 50,
        random_seed: int = 42,
    ) -> SampleExecutionResult:
        """
        Run the pipeline on a sample of the dataset.

        This method should be overridden by subclasses that implement
        actual DJ execution.

        Args:
            cfg: The pipeline configuration
            sample_size: Number of samples to use
            random_seed: Random seed for sampling

        Returns:
            SampleExecutionResult with inputs, outputs, and metrics
        """
        raise NotImplementedError("Subclasses must implement run_sample")

    def run_on_fixed_samples(
        self,
        cfg: DJExecutableConfig,
        samples: List[Dict[str, Any]],
    ) -> SampleExecutionResult:
        """
        Run the pipeline on pre-sampled data.

        This method should be overridden by subclasses that implement
        actual DJ execution.

        Args:
            cfg: The pipeline configuration
            samples: Pre-sampled data to use

        Returns:
            SampleExecutionResult with inputs, outputs, and metrics
        """
        raise NotImplementedError("Subclasses must implement run_on_fixed_samples")


class StubExecutorAdapter(ExecutorAdapter):
    """
    Stub executor for testing without running actual pipelines.

    Generates deterministic fake outputs based on inputs.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
        output_modifier: Optional[callable] = None,
    ) -> None:
        """
        Args:
            dataset_path: Path to input dataset
            ground_truth_path: Path to ground truth labels
            output_modifier: Optional function to modify outputs
        """
        super().__init__(dataset_path, ground_truth_path)
        self._output_modifier = output_modifier

    def run_sample(
        self,
        cfg: DJExecutableConfig,
        sample_size: int = 50,
        random_seed: int = 42,
    ) -> SampleExecutionResult:
        """Generate stub outputs without actual execution."""
        start_time = time.time()

        # Load and sample data
        data = self.load_dataset()
        if not data:
            cfg_path = cfg.get("dataset_path")
            if cfg_path:
                data = self.load_dataset(cfg_path)

        if not data:
            return SampleExecutionResult(
                ok=False,
                errors=["No dataset available"],
                sample_size=0,
            )

        samples = self.sample_data(data, sample_size, random_seed)
        outputs = []

        # Generate fake outputs
        for inp in samples:
            out = dict(inp)
            # Apply simple transformation based on process config
            process = cfg.get("process", [])
            for step in process:
                if not isinstance(step, dict):
                    continue
                op_name = next(iter(step.keys()), "")
                params = step.get(op_name, {})

                # Simulate some operators
                if "filter" in op_name:
                    pass
                elif "mapper" in op_name or "map" in op_name:
                    out["_processed"] = True
                elif "deduplicator" in op_name or "dedup" in op_name:
                    out["_dedup_key"] = hash(str(inp)) % 10000

            if self._output_modifier:
                out = self._output_modifier(out, cfg)

            outputs.append(out)

        ground_truths = self.load_ground_truth() if self._ground_truth_path else []

        token_usage = {
            "prompt_tokens": len(samples) * 100,
            "completion_tokens": len(samples) * 50,
            "model_usage": {
                "gpt-4o-mini": {
                    "prompt": len(samples) * 100,
                    "completion": len(samples) * 50,
                }
            },
        }

        return SampleExecutionResult(
            ok=True,
            inputs=samples,
            outputs=outputs,
            token_usage=token_usage,
            ground_truths=ground_truths[: len(samples)] if ground_truths else [],
            errors=[],
            sample_size=len(samples),
            wall_time_sec=time.time() - start_time,
        )

    def run_on_fixed_samples(
        self,
        cfg: DJExecutableConfig,
        samples: List[Dict[str, Any]],
    ) -> SampleExecutionResult:
        """Generate stub outputs on fixed samples without actual execution."""
        start_time = time.time()
        outputs = []

        for inp in samples:
            out = dict(inp)
            process = cfg.get("process", [])
            for step in process:
                if not isinstance(step, dict):
                    continue
                op_name = next(iter(step.keys()), "")
                params = step.get(op_name, {})

                if "filter" in op_name:
                    pass
                elif "mapper" in op_name or "map" in op_name:
                    out["_processed"] = True
                elif "deduplicator" in op_name or "dedup" in op_name:
                    out["_dedup_key"] = hash(str(inp)) % 10000

            if self._output_modifier:
                out = self._output_modifier(out, cfg)

            outputs.append(out)

        ground_truths = self.load_ground_truth() if self._ground_truth_path else []

        token_usage = {
            "prompt_tokens": len(samples) * 100,
            "completion_tokens": len(samples) * 50,
            "model_usage": {
                "gpt-4o-mini": {
                    "prompt": len(samples) * 100,
                    "completion": len(samples) * 50,
                }
            },
        }

        return SampleExecutionResult(
            ok=True,
            inputs=samples,
            outputs=outputs,
            token_usage=token_usage,
            ground_truths=ground_truths[: len(samples)] if ground_truths else [],
            errors=[],
            sample_size=len(samples),
            wall_time_sec=time.time() - start_time,
        )


class DJExecutorAdapter(ExecutorAdapter):
    """
    Real executor adapter that runs actual Data-Juicer pipelines.

    This adapter interfaces with the DJ executor to run pipelines
    on sample data and collect real metrics.
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
        work_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            dataset_path: Path to input dataset
            ground_truth_path: Path to ground truth labels
            work_dir: Working directory for temporary files
        """
        super().__init__(dataset_path, ground_truth_path)
        self._work_dir = work_dir or "/tmp/dj_optimizer"
        self._token_collector: Optional[TokenUsageCollector] = None

    def set_token_collector(self, collector: "TokenUsageCollector") -> None:
        """Set a token usage collector."""
        self._token_collector = collector

    def run_sample(
        self,
        cfg: DJExecutableConfig,
        sample_size: int = 50,
        random_seed: int = 42,
    ) -> SampleExecutionResult:
        """
        Run the DJ pipeline on a sample.

        This implementation:
        1. Samples the input data
        2. Writes sample to a temporary file
        3. Runs the DJ executor
        4. Collects outputs and token usage
        """
        start_time = time.time()

        try:
            data = self.load_dataset()
            if not data:
                cfg_path = cfg.get("dataset_path")
                if cfg_path:
                    data = self.load_dataset(cfg_path)

            if not data:
                return SampleExecutionResult(
                    ok=False,
                    errors=["No dataset available"],
                    sample_size=0,
                )

            samples = self.sample_data(data, sample_size, random_seed)

            work_path = Path(self._work_dir)
            work_path.mkdir(parents=True, exist_ok=True)

            run_dir = work_path / f"run_{random_seed}_{sample_size}"
            run_dir.mkdir(parents=True, exist_ok=True)

            sample_path = run_dir / "sample.jsonl"
            with sample_path.open("w", encoding="utf-8") as f:
                for item in samples:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            output_path = run_dir / "output.jsonl"

            exec_cfg = dict(cfg)
            exec_cfg["dataset_path"] = str(sample_path)
            exec_cfg["export_path"] = str(output_path)

            outputs, token_usage = self._run_pipeline(exec_cfg, str(run_dir))

            ground_truths = self.load_ground_truth() if self._ground_truth_path else []

            return SampleExecutionResult(
                ok=True,
                inputs=samples,
                outputs=outputs,
                token_usage=token_usage,
                ground_truths=ground_truths[: len(samples)] if ground_truths else [],
                errors=[],
                sample_size=len(samples),
                wall_time_sec=time.time() - start_time,
            )

        except Exception as e:
            return SampleExecutionResult(
                ok=False,
                errors=[str(e)],
                sample_size=0,
                wall_time_sec=time.time() - start_time,
            )

    def run_on_fixed_samples(
        self,
        cfg: DJExecutableConfig,
        samples: List[Dict[str, Any]],
    ) -> SampleExecutionResult:
        """
        Run the DJ pipeline on pre-sampled data.

        This implementation uses fixed samples to ensure consistent
        evaluation throughout the optimization process.
        """
        start_time = time.time()

        try:
            work_path = Path(self._work_dir)
            work_path.mkdir(parents=True, exist_ok=True)

            import uuid

            run_id = str(uuid.uuid4())[:8]
            run_dir = work_path / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            sample_path = run_dir / "sample.jsonl"
            with sample_path.open("w", encoding="utf-8") as f:
                for item in samples:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            output_path = run_dir / "output.jsonl"

            exec_cfg = dict(cfg)
            exec_cfg["dataset_path"] = str(sample_path)
            exec_cfg["export_path"] = str(output_path)

            outputs, token_usage = self._run_pipeline(exec_cfg, str(run_dir))

            ground_truths = self.load_ground_truth() if self._ground_truth_path else []

            return SampleExecutionResult(
                ok=True,
                inputs=samples,
                outputs=outputs,
                token_usage=token_usage,
                ground_truths=ground_truths[: len(samples)] if ground_truths else [],
                errors=[],
                sample_size=len(samples),
                wall_time_sec=time.time() - start_time,
            )

        except Exception as e:
            return SampleExecutionResult(
                ok=False,
                errors=[str(e)],
                sample_size=0,
                wall_time_sec=time.time() - start_time,
            )

    def _run_pipeline(
        self,
        cfg: DJExecutableConfig,
        work_dir: str,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute the pipeline using real Data-Juicer executor.

        Args:
            cfg: Pipeline configuration
            work_dir: Working directory for this run

        Returns:
            Tuple of (outputs list, token_usage dict)
        """
        import sys
        import io
        from contextlib import redirect_stdout, redirect_stderr

        work_dir = os.path.abspath(work_dir)
        config_path = os.path.join(work_dir, "pipeline_config.yaml")
        output_path = cfg.get("export_path", os.path.join(work_dir, "output.jsonl"))

        exec_cfg = dict(cfg)
        exec_cfg["work_dir"] = work_dir
        exec_cfg["export_path"] = output_path
        # Disable DJ monitoring/logging
        exec_cfg["open_monitor"] = False

        with open(config_path, "w", encoding="utf-8") as f:
            import yaml
            yaml.dump(exec_cfg, f, allow_unicode=True, default_flow_style=False)

        try:
            from data_juicer.config import init_configs
            from data_juicer.core import DefaultExecutor

            # Redirect stdout/stderr to suppress DJ logging
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            try:
                args = ["--config", config_path]
                dj_cfg = init_configs(args=args)
                executor = DefaultExecutor(dj_cfg)
                executor.run(skip_return=True)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        except Exception as e:
            print(f"DJ execution error: {e}")
            import traceback
            traceback.print_exc()
            return [], {"prompt_tokens": 0, "completion_tokens": 0, "model_usage": {}}

        outputs = []
        output_path_obj = Path(output_path)
        if output_path_obj.exists():
            with output_path_obj.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        outputs.append(json.loads(line))

        token_usage = self._collect_token_usage(cfg, outputs, work_dir)

        return outputs, token_usage

    def _collect_token_usage(
        self,
        cfg: DJExecutableConfig,
        outputs: List[Dict[str, Any]],
        work_dir: str,
    ) -> Dict[str, Any]:
        """
        Collect token usage from pipeline execution.

        For LLM operators, estimate token usage based on input/output sizes.
        """
        process = cfg.get("process", [])

        llm_ops = {
            "llm_analysis_filter",
            "llm_filter",
            "llm_map",
            "optimize_qa_mapper",
            "optimize_prompt_mapper",
            "generate_qa_from_text_mapper",
            "generate_qa_from_examples_mapper",
            "image_captioning_mapper",
            "mllm_mapper",
            "extract_keyword_mapper",
            "extract_event_mapper",
            "extract_entity_relation_mapper",
            "extract_entity_attribute_mapper",
            "llm_quality_score_filter",
        }

        total_prompt = 0
        total_completion = 0
        model_usage: Dict[str, Dict[str, int]] = {}

        used_llm_ops = []
        for step in process:
            if not isinstance(step, dict):
                continue
            op_name = next(iter(step.keys()), "")
            if op_name in llm_ops:
                params = step.get(op_name, {})
                if isinstance(params, dict):
                    model = params.get("api_model") or params.get("model") or "unknown"
                    used_llm_ops.append((op_name, model))

        if not used_llm_ops:
            return {"prompt_tokens": 0, "completion_tokens": 0, "model_usage": {}}

        for output in outputs:
            text = output.get("text", "")
            output_tokens = len(text) // 4

            for op_name, model in used_llm_ops:
                input_tokens = output_tokens * 2

                total_prompt += input_tokens
                total_completion += output_tokens

                if model not in model_usage:
                    model_usage[model] = {"prompt": 0, "completion": 0}
                model_usage[model]["prompt"] += input_tokens
                model_usage[model]["completion"] += output_tokens

        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "model_usage": model_usage,
        }


@dataclass
class TokenUsageRecord:
    """Record of token usage for a single LLM call."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    timestamp: float = field(default_factory=time.time)


class TokenUsageCollector:
    """
    Collects token usage from LLM calls during pipeline execution.

    This can be hooked into the DJ LLM client to track usage.
    """

    def __init__(self) -> None:
        self._records: List[TokenUsageRecord] = []

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Record a single LLM call's token usage."""
        self._records.append(
            TokenUsageRecord(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all recorded token usage."""
        total_prompt = 0
        total_completion = 0
        model_usage: Dict[str, Dict[str, int]] = {}

        for record in self._records:
            total_prompt += record.prompt_tokens
            total_completion += record.completion_tokens

            if record.model not in model_usage:
                model_usage[record.model] = {"prompt": 0, "completion": 0}
            model_usage[record.model]["prompt"] += record.prompt_tokens
            model_usage[record.model]["completion"] += record.completion_tokens

        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "model_usage": model_usage,
            "call_count": len(self._records),
        }

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()


# Factory functions


def create_stub_adapter(
    dataset_path: Optional[str] = None,
    ground_truth_path: Optional[str] = None,
) -> StubExecutorAdapter:
    """Create a stub executor adapter for testing."""
    return StubExecutorAdapter(dataset_path, ground_truth_path)


def create_real_adapter(
    dataset_path: Optional[str] = None,
    ground_truth_path: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> DJExecutorAdapter:
    """Create a real executor adapter for production use."""
    return DJExecutorAdapter(dataset_path, ground_truth_path, work_dir)
