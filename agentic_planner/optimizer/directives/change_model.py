# -*- coding: utf-8 -*-
"""
Model change directive for LLM-based operators.

This module provides SwapSingleOpModelDirective for replacing the model
of a specific operator. This is the only model-change directive that
complies with the Action-based design (one operator, one directive).
"""

from __future__ import annotations

from typing import Optional, Tuple

from agentic_planner.contracts.recipe import DJExecutableConfig
from agentic_planner.optimizer.directives.base import Directive, DirectiveResult
from agentic_planner.optimizer.model_registry import ModelRegistry
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

    name = "swap_model"
    applicable_op_types = None

    def __init__(
        self,
        locator: Optional[OpLocator] = None,
        from_model: str = "",
        to_model: str = "",
        model_keys: tuple[str, ...] = ("api_model", "model"),
        model_registry: Optional[ModelRegistry] = None,
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
        self.model_registry = model_registry or ModelRegistry.default()

    def _resolve_model_key_and_value(self, params: dict) -> Tuple[Optional[str], str]:
        """Resolve the active model parameter key and current value."""
        for key in self.model_keys:
            value = params.get(key)
            if isinstance(value, str) and value:
                return key, value
        return None, ""

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

        model_key, current_model = self._resolve_model_key_and_value(params)
        if model_key is None:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="operator has no model parameter",
                config_before=before,
                config_after=after,
            )

        if self.from_model and current_model != self.from_model:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message=f"current model {current_model} does not match {self.from_model}",
                config_before=before,
                config_after=after,
            )

        if not self.to_model:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="target model is required",
                config_before=before,
                config_after=after,
            )

        if not self.model_registry.is_swap_compatible(current_model, self.to_model):
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message=f"model swap blocked by compatibility checks: {current_model} -> {self.to_model}",
                config_before=before,
                config_after=after,
            )

        params[model_key] = self.to_model

        after["process"][target_idx] = {op_name: params}

        identity = index.get_by_index(target_idx)

        return DirectiveResult(
            ok=True,
            applied=True,
            directive_name=self.name,
            message=f"{current_model} -> {self.to_model}",
            config_before=before,
            config_after=after,
            details={
                "action_type": "model_swap",
                "identity_hash": identity.identity_hash if identity else None,
                "operator_id": identity.operator_id if identity else None,
                "op_type": op_name,
                "from_model": current_model,
                "to_model": self.to_model,
                "model_param_key": model_key,
            },
        )


__all__ = ["SwapSingleOpModelDirective"]
