# -*- coding: utf-8 -*-
"""Directive specification templates and metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from agentic_planner.optimizer.directives.base import Directive

if TYPE_CHECKING:
    from agentic_planner.optimizer.directives.instances import InstantiatedDirective
    from agentic_planner.optimizer.op_locator import TargetLocator


@dataclass(frozen=True)
class DirectiveApplicability:
    """Applicability metadata describing where a directive may run."""

    per_operator: bool = True
    """Whether this directive can be applied to a specific operator target."""

    global_allowed: bool = True
    """Whether this directive can run without a specific operator target."""

    applicable_op_types: List[str] = field(default_factory=list)
    """Operator types eligible for application. Empty means any type."""

    target_locator_supported: bool = True
    """Whether canonical target locator resolution is supported."""


@dataclass(frozen=True)
class DirectiveSpec:
    """Template contract for constructing directive instances."""

    name: str
    directive_factory: Callable[..., Directive]
    default_params: Dict[str, Any] = field(default_factory=dict)
    safety_level: str = "safe"
    safety_notes: str = ""
    applicability: DirectiveApplicability = field(default_factory=DirectiveApplicability)

    def instantiate(
        self,
        params: Optional[Dict[str, Any]] = None,
        target_locator: Optional["TargetLocator"] = None,
    ) -> "InstantiatedDirective":
        """Build an instantiated directive with resolved runtime parameters."""
        from agentic_planner.optimizer.directives.instances import InstantiatedDirective

        merged_params = dict(self.default_params)
        if params:
            merged_params.update(params)

        directive = self.directive_factory(**merged_params)
        return InstantiatedDirective(
            spec_name=self.name,
            directive=directive,
            params=merged_params,
            target_locator=target_locator,
            safety_level=self.safety_level,
            applicability=self.applicability,
        )
