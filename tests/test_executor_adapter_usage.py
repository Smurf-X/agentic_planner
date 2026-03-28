# -*- coding: utf-8 -*-
"""Token usage collection behavior for DJExecutorAdapter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agentic_planner.optimizer.executor_adapter import DJExecutorAdapter


def test_collect_token_usage_does_not_estimate_from_output_text() -> None:
    adapter = DJExecutorAdapter(dataset_path=None)
    cfg = {
        "process": [
            {
                "llm_task_relevance_filter": {
                    "api_or_hf_model": "qwen3-8b",
                    "task_desc": "demo",
                }
            }
        ]
    }
    outputs = [{"text": "this looks like a normal output payload"}]

    empty_dir = tempfile.mkdtemp(prefix="ap_usage_empty_")
    usage = adapter._collect_token_usage(cfg, outputs, work_dir=empty_dir)

    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["model_usage"] == {}


def test_collect_token_usage_reads_explicit_usage_from_outputs() -> None:
    adapter = DJExecutorAdapter(dataset_path=None)
    cfg = {
        "process": [
            {
                "llm_task_relevance_filter": {
                    "api_or_hf_model": "qwen3-8b",
                    "task_desc": "demo",
                }
            }
        ]
    }
    outputs = [{"text": "x"}, {"text": "y"}]
    work_dir = Path("/tmp/ap_usage_test")
    work_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "model_usage": {
            "qwen3-8b": {"prompt": 20, "completion": 5},
        },
    }
    with (work_dir / "llm_usage_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f)

    usage = adapter._collect_token_usage(cfg, outputs, work_dir=str(work_dir))

    assert usage["prompt_tokens"] == 20
    assert usage["completion_tokens"] == 5
    assert usage["model_usage"] == {"qwen3-8b": {"prompt": 20, "completion": 5}}
