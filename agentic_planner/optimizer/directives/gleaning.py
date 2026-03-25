# -*- coding: utf-8 -*-
"""Add gleaning (multi-turn verification) to LLM-based operators."""

from __future__ import annotations

from typing import Any, Dict, Optional

from agentic_planner.contracts.recipe import DJExecutableConfig
from agentic_planner.optimizer.directives.base import Directive, DirectiveResult
from agentic_planner.optimizer.op_locator import OpLocator, ProcessIndex


# Operators that support gleaning
_GLEANING_SUPPORTED_OPS = {
    "map_wrapper",
    "llm_map",
    "filter_wrapper",
    "llm_filter",
}


class AddGleaningDirective(Directive):
    """
    Add gleaning parameters to an LLM-based operator.

    Gleaning enables multi-turn verification where the LLM can
    refine its output through additional passes.

    Uses OpLocator for stable identification.
    """

    name = "add_gleaning"

    def __init__(
        self,
        locator: OpLocator,
        max_rounds: int = 2,
        gleaning_prompt: Optional[str] = None,
    ) -> None:
        """
        Args:
            locator: Locator for the target operator
            max_rounds: Maximum number of gleaning rounds (default: 2)
            gleaning_prompt: Custom prompt for gleaning verification
        """
        self.locator = locator
        self.max_rounds = max(1, min(max_rounds, 5))
        self.gleaning_prompt = gleaning_prompt

    def apply_with_index(
        self,
        cfg: DJExecutableConfig,
        index: ProcessIndex,
    ) -> DirectiveResult:
        before = self._clone(cfg)

        # Find target operator
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

        # Check if gleaning is supported for this operator
        if op_name not in _GLEANING_SUPPORTED_OPS:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message=f"gleaning not supported for {op_name}",
                config_before=before,
                config_after=after,
            )

        # Check if gleaning already enabled
        if params.get("max_rounds", 1) > 1:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="gleaning already enabled",
                config_before=before,
                config_after=after,
            )

        new_params = dict(params)
        new_params["max_rounds"] = self.max_rounds

        if self.gleaning_prompt:
            new_params["gleaning_prompt"] = self.gleaning_prompt

        after["process"][target_idx] = {op_name: new_params}

        identity = index.get_by_index(target_idx)

        return DirectiveResult(
            ok=True,
            applied=True,
            directive_name=self.name,
            message=f"added gleaning ({self.max_rounds} rounds) to {op_name}",
            config_before=before,
            config_after=after,
            details={
                "identity_hash": identity.identity_hash if identity else None,
                "op_type": op_name,
                "max_rounds": self.max_rounds,
            },
        )


class RemoveGleaningDirective(Directive):
    """
    Remove gleaning from an LLM-based operator (for cost optimization).

    Uses OpLocator for stable identification.
    """

    name = "remove_gleaning"

    def __init__(self, locator: OpLocator) -> None:
        """
        Args:
            locator: Locator for the target operator
        """
        self.locator = locator

    def apply_with_index(
        self,
        cfg: DJExecutableConfig,
        index: ProcessIndex,
    ) -> DirectiveResult:
        before = self._clone(cfg)

        # Find target operator
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

        if params.get("max_rounds", 1) <= 1:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="gleaning not enabled",
                config_before=before,
                config_after=after,
            )

        new_params = dict(params)
        new_params["max_rounds"] = 1
        new_params.pop("gleaning_prompt", None)

        after["process"][target_idx] = {op_name: new_params}

        identity = index.get_by_index(target_idx)

        return DirectiveResult(
            ok=True,
            applied=True,
            directive_name=self.name,
            message=f"removed gleaning from {op_name}",
            config_before=before,
            config_after=after,
            details={
                "identity_hash": identity.identity_hash if identity else None,
                "op_type": op_name,
            },
        )