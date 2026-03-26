# -*- coding: utf-8 -*-
"""
Model change directive for LLM-based operators.

This module provides SwapSingleOpModelDirective for replacing the model
of a specific operator. This is the only model-change directive that
complies with the Action-based design (one operator, one directive).
"""

from __future__ import annotations

from typing import Optional

from agentic_planner.contracts.recipe import DJExecutableConfig
from agentic_planner.optimizer.directives.base import Directive, DirectiveResult
from agentic_planner.optimizer.op_locator import OpLocator, ProcessIndex


class SwapSingleOpModelDirective(Directive):
    """
    Replace model for a specific operator.

    This directive complies with Action-based design: it only affects
    the operator specified by target_op or locator, not all operators.

    Example:
        directive = SwapSingleOpModelDirective(
            locator=OpLocator(op_type="llm_quality_score_filter", occurrence=0),
            from_model="gpt-4o-mini",
            to_model="gpt-4o",
        )
    """

    name = "swap_single_op_model"
    applicable_op_types = None

    def __init__(
        self,
        locator: Optional[OpLocator] = None,
        from_model: str = "",
        to_model: str = "",
        model_keys: tuple[str, ...] = ("api_model", "model"),
    ) -> None:
        """
        Args:
            locator: Locator for the target operator (optional if target_op is used)
            from_model: Source model name (only replace if matches)
            to_model: Target model name
            model_keys: Parameter keys that contain model name
        """
        self.locator = locator
        self.from_model = from_model
        self.to_model = to_model
        self.model_keys = model_keys

    def apply_with_index(
        self,
        cfg: DJExecutableConfig,
        index: ProcessIndex,
        target_op: Optional[int] = None,
    ) -> DirectiveResult:
        before = self._clone(cfg)

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
                config_after=after,
            )

        changed = False
        for key in self.model_keys:
            if key in params and params[key] == self.from_model:
                params[key] = self.to_model
                changed = True

        after["process"][target_idx] = {op_name: params}

        identity = index.get_by_index(target_idx)

        return DirectiveResult(
            ok=True,
            applied=changed,
            directive_name=self.name,
            message=f"{self.from_model} -> {self.to_model}" if changed else "no matching model",
            config_before=before,
            config_after=after,
            details={
                "identity_hash": identity.identity_hash if identity else None,
                "op_type": op_name,
            },
        )


__all__ = ["SwapSingleOpModelDirective"]
