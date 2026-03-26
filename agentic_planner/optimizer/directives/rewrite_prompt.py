# -*- coding: utf-8 -*-
"""LLM-driven prompt rewriting directive for LLM-based operators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agentic_planner.contracts.recipe import DJExecutableConfig
from agentic_planner.optimizer.directives.base import Directive, DirectiveResult
from agentic_planner.optimizer.op_locator import OpLocator, ProcessIndex

if TYPE_CHECKING:
    pass


# Operators that have a 'prompt' parameter
_LLM_PROMPT_OPS = {
    "map_wrapper": "prompt",
    "filter_wrapper": "prompt",
    "llm_map": "prompt",
    "llm_filter": "prompt",
}


class RewritePromptDirective(Directive):
    """
    Rewrite the prompt of an LLM-based operator.

    Uses OpLocator for stable identification across pipeline transformations.

    Example:
        # Rewrite prompt for an llm_filter with specific content
        locator = OpLocator(op_type="llm_filter", param_match={"prompt": "contains:summarize"})
        directive = RewritePromptDirective(locator, prompt_suffix="Be concise.")
    """

    name = "rewrite_prompt"
    applicable_op_types = list(_LLM_PROMPT_OPS.keys())

    def __init__(
        self,
        locator: OpLocator,
        new_prompt: Optional[str] = None,
        prompt_suffix: Optional[str] = None,
        clarify_instruction: Optional[str] = None,
    ) -> None:
        """
        Args:
            locator: Locator for the target operator
            new_prompt: Replace the entire prompt with this
            prompt_suffix: Append this to the existing prompt
            clarify_instruction: Prepend clarification to the prompt
        """
        self.locator = locator
        self.new_prompt = new_prompt
        self.prompt_suffix = prompt_suffix
        self.clarify_instruction = clarify_instruction

    def apply_with_index(
        self,
        cfg: DJExecutableConfig,
        index: ProcessIndex,
        target_op: Optional[int] = None,
    ) -> DirectiveResult:
        before = self._clone(cfg)

        target_idx = target_op
        if target_idx is None:
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
        if prompt_key not in params:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message=f"operator {op_name} has no prompt parameter",
                config_before=before,
                config_after=after,
            )

        old_prompt = params.get(prompt_key, "")
        new_params = dict(params)

        if self.new_prompt:
            new_params[prompt_key] = self.new_prompt
        elif self.prompt_suffix:
            new_params[prompt_key] = old_prompt + "\n\n" + self.prompt_suffix
        elif self.clarify_instruction:
            new_params[prompt_key] = f"[Instruction: {self.clarify_instruction}]\n\n{old_prompt}"
        else:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="no transformation specified",
                config_before=before,
                config_after=after,
            )

        after["process"][target_idx] = {op_name: new_params}

        identity = index.get_by_index(target_idx)

        return DirectiveResult(
            ok=True,
            applied=True,
            directive_name=self.name,
            message=f"rewrote prompt for {op_name}",
            config_before=before,
            config_after=after,
            details={
                "identity_hash": identity.identity_hash if identity else None,
                "op_type": op_name,
                "old_prompt_preview": old_prompt[:100] + "..."
                if len(old_prompt) > 100
                else old_prompt,
            },
        )


class AddFewShotExamplesDirective(Directive):
    """
    Add few-shot examples to an LLM operator's prompt.

    Uses OpLocator for stable identification.
    """

    name = "add_few_shot_examples"
    applicable_op_types = list(_LLM_PROMPT_OPS.keys())

    def __init__(self, locator: OpLocator, examples: List[Dict[str, str]]) -> None:
        """
        Args:
            locator: Locator for the target operator
            examples: List of example dicts with 'input' and 'output' keys
        """
        self.locator = locator
        self.examples = examples

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
        if target_idx is None:
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
        old_prompt = params.get(prompt_key, "")
        new_params = dict(params)

        examples_text = "\n\n## Examples:\n"
        for i, ex in enumerate(self.examples, 1):
            inp = ex.get("input", "")
            out = ex.get("output", "")
            examples_text += f"\n### Example {i}:\n"
            examples_text += f"Input: {inp}\n"
            examples_text += f"Output: {out}\n"

        new_params[prompt_key] = old_prompt + examples_text
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
                "identity_hash": identity.identity_hash if identity else None,
                "examples_count": len(self.examples),
            },
        )
