# -*- coding: utf-8 -*-
"""Base classes for optimization directives."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from agentic_planner.contracts.recipe import DJExecutableConfig

if TYPE_CHECKING:
    from agentic_planner.optimizer.op_locator import OpLocator, ProcessIndex


@dataclass
class DirectiveResult:
    """
    Result of a directive application.

    Captures the outcome of applying a directive to a configuration,
    including before/after states for auditability.
    """

    ok: bool
    """Whether the directive executed without errors."""

    applied: bool
    """Whether the directive actually modified the config."""

    directive_name: str
    """Name of the directive that produced this result."""

    message: str = ""
    """Human-readable description of what happened."""

    config_before: Optional[Dict[str, Any]] = None
    """Config state before the directive."""

    config_after: Optional[Dict[str, Any]] = None
    """Config state after the directive."""

    details: Dict[str, Any] = field(default_factory=dict)
    """Additional details about the transformation."""


class Directive(ABC):
    """
    Base class for optimization directives.

    A directive is a single-step transformation of a pipeline configuration.
    Directives are the atomic actions used by both:
    - DirectiveEngine for sequential optimization
    - Search strategies for exploring the configuration space

    Subclasses must implement apply_with_index() which receives the config
    and a ProcessIndex for stable operator location.

    Use OpLocator instead of position indices to identify target operators,
    as indices become invalid when the pipeline structure changes.

    Attributes:
        name: Unique name for this directive
        applicable_op_types: List of operator types this directive applies to.
                             None means applicable to all operators.
    """

    name: str = "base"
    applicable_op_types: Optional[List[str]] = None

    def apply(
        self,
        cfg: DJExecutableConfig,
        target_op: Optional[int] = None,
    ) -> DirectiveResult:
        """
        Apply the directive to a configuration.

        Args:
            cfg: The pipeline configuration to transform
            target_op: Index of target operator (None for global application)

        Returns:
            DirectiveResult with the outcome
        """
        from agentic_planner.optimizer.op_locator import ProcessIndex

        index = ProcessIndex.build(cfg.get("process", []))
        return self.apply_with_index(cfg, index, target_op=target_op)

    @abstractmethod
    def apply_with_index(
        self,
        cfg: DJExecutableConfig,
        index: "ProcessIndex",
        target_op: Optional[int] = None,
    ) -> DirectiveResult:
        """
        Apply the directive with a pre-built index.

        Args:
            cfg: The pipeline configuration to transform
            index: ProcessIndex for stable operator location
            target_op: Index of target operator (None for global application)

        Returns:
            DirectiveResult with the outcome
        """
        pass

    def is_applicable(
        self,
        op_type: str,
        params: Dict[str, Any],
        index: "ProcessIndex",
    ) -> bool:
        """
        Check if this directive is applicable to an operator.

        Args:
            op_type: Operator type name
            params: Operator parameters
            index: Process index for context

        Returns:
            True if the directive can be applied to this operator
        """
        if self.applicable_op_types is not None:
            return op_type in self.applicable_op_types
        return True

    def _clone(self, cfg: DJExecutableConfig) -> DJExecutableConfig:
        """Create a deep copy of the config."""
        return deepcopy(cfg)

    def _get_step(self, cfg: DJExecutableConfig, idx: int) -> Optional[Dict[str, Any]]:
        """Get a step from the process list by index."""
        process = cfg.get("process", [])
        if 0 <= idx < len(process):
            return process[idx]
        return None

    def _get_op_params(
        self,
        step: Dict[str, Any],
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Extract operator name and params from a step.

        Returns:
            Tuple of (op_name, params_dict)
        """
        if not isinstance(step, dict) or len(step) != 1:
            return None, {}
        op_name = next(iter(step.keys()))
        params = step.get(op_name, {})
        if not isinstance(params, dict):
            params = {}
        return op_name, params
