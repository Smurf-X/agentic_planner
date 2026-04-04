# -*- coding: utf-8 -*-
"""Few-shot prompt example directives for LLM-based operators."""

from __future__ import annotations

from typing import Dict, List, Optional

from agentic_planner.contracts.recipe import DJExecutableConfig
from agentic_planner.optimizer.directives.base import Directive, DirectiveResult
from agentic_planner.optimizer.op_locator import OpLocator, ProcessIndex


_LLM_PROMPT_OPS = {
    "map_wrapper": "prompt",
    "filter_wrapper": "prompt",
    "llm_map": "prompt",
    "llm_filter": "prompt",
}


class AddFewShotExamplesDirective(Directive):
    """Add few-shot examples to an LLM operator prompt."""

    name = "add_few_shot_examples"
    applicable_op_types = list(_LLM_PROMPT_OPS.keys())

    def __init__(
        self,
        locator: Optional[OpLocator] = None,
        examples: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        self.locator = locator
        self.examples = examples or []

    def apply_with_index(
        self,
        cfg: DJExecutableConfig,
        index: ProcessIndex,
        target_op: Optional[int] = None,
    ) -> DirectiveResult:
        before = self._clone(cfg)

        if not self.examples:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="no examples provided",
                config_before=before,
                config_after=before,
            )

        target_idx = target_op
        if target_idx is None and self.locator is not None:
            target_idx = self.locator.find_index(index.identities)
        if target_idx is None:
            return DirectiveResult(
                ok=False,
                applied=False,
                directive_name=self.name,
                message="target operator not found",
                config_before=before,
                config_after=before,
            )

        after = self._clone(before)
        step = after["process"][target_idx]

        op_name, params = self._get_op_params(step)
        if op_name is None:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="invalid step format",
                config_before=before,
                config_after=before,
            )

        prompt_key = _LLM_PROMPT_OPS.get(op_name, "prompt")
        old_prompt = str(params.get(prompt_key, ""))
        new_params = dict(params)

        examples_text = "\n\n## Examples:\n"
        for i, ex in enumerate(self.examples, 1):
            inp = ex.get("input", "")
            out = ex.get("output", "")
            examples_text += f"\n### Example {i}:\n"
            examples_text += f"Input: {inp}\n"
            examples_text += f"Output: {out}\n"

        new_prompt = old_prompt + examples_text
        new_params[prompt_key] = new_prompt
        after["process"][target_idx] = {op_name: new_params}

        identity = index.get_by_index(target_idx)

        return DirectiveResult(
            ok=True,
            applied=True,
            directive_name=self.name,
            message=f"added {len(self.examples)} example(s) to {op_name}",
            config_before=before,
            config_after=after,
            details={
                "action_type": "few_shot_examples",
                "identity_hash": identity.identity_hash if identity else None,
                "operator_id": identity.operator_id if identity else None,
                "op_type": op_name,
                "prompt_key": prompt_key,
                "examples_added": len(self.examples),
                "prompt_before_chars": len(old_prompt),
                "prompt_after_chars": len(new_prompt),
            },
        )


__all__ = ["AddFewShotExamplesDirective"]
