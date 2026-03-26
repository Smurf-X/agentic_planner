# -*- coding: utf-8 -*-
"""Registry for optimization directives."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentic_planner.optimizer.directives.base import Directive
from agentic_planner.optimizer.directives.adjust_params import BumpMinLenDirective
from agentic_planner.optimizer.directives.adjust_threshold import (
    AdjustThresholdDirective,
    LoosenFiltersDirective,
    TightenFiltersDirective,
)
from agentic_planner.optimizer.directives.remove_redundant import RemoveRedundantOpDirective
from agentic_planner.optimizer.directives.reorder import ReorderFiltersFirstDirective
from agentic_planner.optimizer.directives.specs import DirectiveApplicability, DirectiveSpec

# Core registry with singleton instances
DIRECTIVE_REGISTRY: Dict[str, Directive] = {}
DIRECTIVE_SPEC_REGISTRY: Dict[str, DirectiveSpec] = {}

_CORE_DIRECTIVE_NAMES = {
    "tighten_threshold",
    "loosen_threshold",
    "remove_redundant_op",
    "safe_reorder_local",
    "reorder_filters_first",
    "remove_redundant_ops",
    "tighten_filters",
    "loosen_filters",
    "bump_text_length_min_len",
}


def _register_instance(d: Directive) -> None:
    """Register a directive instance under its name."""
    DIRECTIVE_REGISTRY[d.name] = d


def _register_spec(spec: DirectiveSpec) -> None:
    """Register a directive spec and its default runtime instance."""
    DIRECTIVE_SPEC_REGISTRY[spec.name] = spec
    DIRECTIVE_REGISTRY[spec.name] = spec.instantiate().directive


def _register_default_specs() -> None:
    """Register first-wave directive specs with metadata."""
    _register_spec(
        DirectiveSpec(
            name="tighten_threshold",
            directive_factory=TightenFiltersDirective,
            default_params={"intensity": 0.1},
            safety_level="safe",
            safety_notes="Monotonic threshold tightening on known numeric filter fields.",
            applicability=DirectiveApplicability(
                per_operator=True,
                global_allowed=True,
                applicable_op_types=list(TightenFiltersDirective.applicable_op_types),
                target_locator_supported=True,
            ),
        )
    )
    _register_spec(
        DirectiveSpec(
            name="loosen_threshold",
            directive_factory=LoosenFiltersDirective,
            default_params={"intensity": 0.1},
            safety_level="safe",
            safety_notes="Reverse threshold adjustments while preserving non-negative bounds.",
            applicability=DirectiveApplicability(
                per_operator=True,
                global_allowed=True,
                applicable_op_types=list(LoosenFiltersDirective.applicable_op_types),
                target_locator_supported=True,
            ),
        )
    )
    _register_spec(
        DirectiveSpec(
            name="remove_redundant_op",
            directive_factory=RemoveRedundantOpDirective,
            default_params={"remove_duplicates": True, "remove_noops": True},
            safety_level="safe",
            safety_notes="Only removes duplicate or provable no-op process steps.",
            applicability=DirectiveApplicability(
                per_operator=False,
                global_allowed=True,
                applicable_op_types=[],
                target_locator_supported=False,
            ),
        )
    )
    _register_spec(
        DirectiveSpec(
            name="safe_reorder_local",
            directive_factory=ReorderFiltersFirstDirective,
            default_params={"local_only": True},
            safety_level="safe",
            safety_notes="Uses adjacent local swaps to avoid broad pipeline reshuffling.",
            applicability=DirectiveApplicability(
                per_operator=False,
                global_allowed=True,
                applicable_op_types=[],
                target_locator_supported=False,
            ),
        )
    )


# Register default directives
_register_default_specs()

# Legacy names retained for backward compatibility.
DIRECTIVE_REGISTRY["tighten_filters"] = DIRECTIVE_REGISTRY["tighten_threshold"]
DIRECTIVE_REGISTRY["loosen_filters"] = DIRECTIVE_REGISTRY["loosen_threshold"]
DIRECTIVE_REGISTRY["remove_redundant_ops"] = DIRECTIVE_REGISTRY["remove_redundant_op"]
_register_instance(ReorderFiltersFirstDirective(local_only=False))
_register_instance(BumpMinLenDirective(10))


def register_directive(directive: Directive, name: Optional[str] = None) -> str:
    """
    Register a custom directive instance.

    Args:
        directive: The directive instance to register
        name: Optional custom name (default: use directive.name)

    Returns:
        The name under which the directive was registered
    """
    key = name or directive.name
    DIRECTIVE_REGISTRY[key] = directive
    return key


def get_directive(name: str) -> Optional[Directive]:
    """Get a directive by name."""
    return DIRECTIVE_REGISTRY.get(name)


def register_directive_spec(spec: DirectiveSpec, name: Optional[str] = None) -> str:
    """Register a custom directive specification."""
    key = name or spec.name
    if key != spec.name:
        spec = DirectiveSpec(
            name=key,
            directive_factory=spec.directive_factory,
            default_params=dict(spec.default_params),
            safety_level=spec.safety_level,
            safety_notes=spec.safety_notes,
            applicability=spec.applicability,
        )
    DIRECTIVE_SPEC_REGISTRY[key] = spec
    DIRECTIVE_REGISTRY[key] = spec.instantiate().directive
    return key


def get_directive_spec(name: str) -> Optional[DirectiveSpec]:
    """Get a directive specification by name."""
    return DIRECTIVE_SPEC_REGISTRY.get(name)


def list_directive_spec_names() -> List[str]:
    """List all registered directive specification names."""
    return sorted(DIRECTIVE_SPEC_REGISTRY.keys())


def list_directive_names() -> List[str]:
    """List all registered directive names."""
    return sorted(DIRECTIVE_REGISTRY.keys())


def clear_dynamic_directives() -> None:
    """
    Clear dynamically registered directives (keep core ones).

    Useful for testing or resetting state.
    """
    to_remove = [k for k in DIRECTIVE_REGISTRY if k not in _CORE_DIRECTIVE_NAMES]
    for k in to_remove:
        del DIRECTIVE_REGISTRY[k]

    spec_to_remove = [k for k in DIRECTIVE_SPEC_REGISTRY if k not in _CORE_DIRECTIVE_NAMES]
    for k in spec_to_remove:
        del DIRECTIVE_SPEC_REGISTRY[k]


# Convenience registration functions

def register_threshold_directive(
    op_type: str,
    param_name: str,
    delta: float,
    direction: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """Register a parameterized threshold adjustment directive."""
    d = AdjustThresholdDirective(op_type, param_name, delta, direction)
    key = name or f"adjust_{op_type}_{param_name}"
    DIRECTIVE_REGISTRY[key] = d
    return key
