# -*- coding: utf-8 -*-
"""
Action and ActionSpace for pipeline optimization.

This module provides a unified abstraction for optimization actions,
decoupling action construction from search strategies.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from agentic_planner.optimizer.op_locator import ProcessIndex, TargetLocator

if TYPE_CHECKING:
    from agentic_planner.contracts.recipe import DJExecutableConfig
    from agentic_planner.optimizer.directives.base import Directive, DirectiveResult
    from agentic_planner.optimizer.directives.specs import DirectiveSpec
    from agentic_planner.optimizer.model_registry import ModelRegistry
    from agentic_planner.optimizer.op_locator import OpIdentity


def _directive_signature(directive: "Directive") -> str:
    """Generate a stable directive instance signature."""
    if hasattr(directive, "replay_signature"):
        return directive.replay_signature()

    payload = {
        "directive_class": directive.__class__.__name__,
        "state": getattr(directive, "__dict__", {}),
    }
    content = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


_DIRECTIVE_TEMPLATE_ALIASES: Dict[str, str] = {
    "tighten_filters": "tighten_threshold",
    "loosen_filters": "loosen_threshold",
    "remove_redundant_ops": "remove_redundant_op",
    "reorder_filters_first": "safe_reorder_local",
}


def resolve_directive_template_name(directive: "Directive", directive_name: Optional[str] = None) -> str:
    """Resolve canonical directive template name for a directive instance."""
    candidate = getattr(directive, "spec_name", None)
    if isinstance(candidate, str) and candidate:
        return candidate

    effective_name = directive_name or directive.name
    return _DIRECTIVE_TEMPLATE_ALIASES.get(effective_name, effective_name)


def _json_safe(value: Any) -> Any:
    """Convert values to JSON-safe forms for planning payloads."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            return str(value)
    return str(value)


def extract_directive_instantiate_params(directive: "Directive") -> Dict[str, Any]:
    """Extract directive instance state as instantiate params."""
    state = getattr(directive, "__dict__", {})
    if not isinstance(state, dict):
        return {}

    params: Dict[str, Any] = {}
    for key, value in state.items():
        if key.startswith("_"):
            continue
        params[str(key)] = _json_safe(value)
    return params


@dataclass
class Action:
    """
    Unified action abstraction for optimization.

    An action represents applying a directive to a specific operator
    in the pipeline. This is the atomic unit used by search strategies.

    Attributes:
        target_locator: Canonical locator for the target operator
        operator_name: Name/type of the target operator
        directive: The directive to apply
        directive_name: Name of the directive
        directive_signature: Stable signature of directive instance parameters
    """

    target_locator: TargetLocator
    operator_name: str
    directive: Directive
    directive_name: str = ""
    directive_template: str = ""
    instantiate_params: Dict[str, Any] = field(default_factory=dict)
    directive_signature: str = ""
    operator_identity: Optional[OpIdentity] = None

    def __post_init__(self):
        if not self.directive_name:
            self.directive_name = self.directive.name
        if not self.directive_template:
            self.directive_template = resolve_directive_template_name(
                self.directive,
                self.directive_name,
            )
        if not self.instantiate_params:
            self.instantiate_params = extract_directive_instantiate_params(self.directive)
        if not self.directive_signature:
            self.directive_signature = _directive_signature(self.directive)

    @property
    def action_key(self) -> str:
        """Stable identity for deduplicating actions."""
        return f"{self.target_locator.canonical_key()}:{self.directive_name}:{self.directive_signature}"

    def resolve_target_index(self, index: ProcessIndex) -> Optional[int]:
        """Resolve the target in the current process index."""
        return index.locate_target(self.target_locator)

    def apply(self, config: DJExecutableConfig) -> DirectiveResult:
        """
        Apply the directive to the target operator.

        Args:
            config: Pipeline configuration

        Returns:
            DirectiveResult with outcome
        """
        index = ProcessIndex.build(config.get("process", []))
        target_idx = self.resolve_target_index(index)
        if target_idx is None:
            from agentic_planner.optimizer.directives.base import DirectiveResult

            return DirectiveResult(
                ok=False,
                applied=False,
                directive_name=self.directive_name,
                message=(
                    "target operator no longer exists "
                    f"(operator_id={self.target_locator.operator_id})"
                ),
                config_before=config,
                config_after=config,
                details={
                    "invalid_target": True,
                    "target_locator": self.target_locator.to_dict(),
                    "action_key": self.action_key,
                },
            )

        return self.directive.apply(config, target_op=target_idx)

    def __hash__(self) -> int:
        return hash(self.action_key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Action):
            return False
        return self.action_key == other.action_key

    def to_dict(self) -> Dict[str, Any]:
        """Serialize action to dict for logging/debugging."""
        return {
            "target_locator": self.target_locator.to_dict(),
            "operator_name": self.operator_name,
            "directive_name": self.directive_name,
            "directive_template": self.directive_template,
            "instantiate_params": self.instantiate_params,
            "directive_signature": self.directive_signature,
            "action_key": self.action_key,
        }

    def __repr__(self) -> str:
        return (
            f"Action({self.operator_name}[{self.target_locator.operator_id}] "
            f"-> {self.directive_name})"
        )


@dataclass
class ActionSpace:
    """
    Collection of available actions for a pipeline configuration.

    ActionSpace is built from a pipeline config and a set of directives,
    representing all possible (operator, directive) combinations.

    Attributes:
        actions: List of available actions
        config_hash: Hash of the source config (for caching)
        operator_count: Number of operators in the pipeline
    """

    actions: List[Action] = field(default_factory=list)
    config_hash: str = ""
    operator_count: int = 0

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    def __getitem__(self, index: int) -> Action:
        return self.actions[index]

    def filter(self, predicate: Callable[[Action], bool]) -> ActionSpace:
        """
        Filter actions by a predicate.

        Args:
            predicate: Function that returns True for actions to keep

        Returns:
            New ActionSpace with filtered actions
        """
        filtered = [a for a in self.actions if predicate(a)]
        return ActionSpace(
            actions=filtered,
            config_hash=self.config_hash,
            operator_count=self.operator_count,
        )

    def exclude(self, used_actions: set) -> ActionSpace:
        """
        Exclude already-used actions.

        Args:
            used_actions: Set of action keys

        Returns:
            New ActionSpace without used actions
        """
        return self.filter(lambda a: a.action_key not in used_actions)

    def get_for_operator(self, operator_id: str) -> List[Action]:
        """
        Get all actions targeting a specific operator.

        Args:
            operator_id: Stable id of the operator

        Returns:
            List of actions for that operator
        """
        return [a for a in self.actions if a.target_locator.operator_id == operator_id]

    def get_for_directive(self, directive_name: str) -> List[Action]:
        """
        Get all actions using a specific directive.

        Args:
            directive_name: Name of the directive

        Returns:
            List of actions using that directive
        """
        return [a for a in self.actions if a.directive_name == directive_name]

    def sample(self, n: int, rng: Any = None) -> List[Action]:
        """
        Sample n random actions.

        Args:
            n: Number of actions to sample
            rng: Random generator (optional)

        Returns:
            List of sampled actions
        """
        import random

        gen = rng if rng else random.Random()
        if n >= len(self.actions):
            return self.actions.copy()
        return gen.sample(self.actions, n)

    def group_by_operator(self) -> Dict[str, List[Action]]:
        """
        Group actions by target operator.

        Returns:
            Dict mapping operator_id to list of actions
        """
        groups: Dict[str, List[Action]] = {}
        for action in self.actions:
            operator_id = action.target_locator.operator_id
            if operator_id not in groups:
                groups[operator_id] = []
            groups[operator_id].append(action)
        return groups

    def to_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics.

        Returns:
            Dict with action space statistics
        """
        by_op = self.group_by_operator()
        by_directive: Dict[str, int] = {}
        for action in self.actions:
            by_directive[action.directive_name] = by_directive.get(action.directive_name, 0) + 1

        return {
            "total_actions": len(self.actions),
            "operator_count": self.operator_count,
            "actions_per_operator": {k: len(v) for k, v in by_op.items()},
            "actions_by_directive": by_directive,
        }

    @staticmethod
    def hash_config(config: Dict[str, Any]) -> str:
        """Generate a hash for a configuration."""
        content = json.dumps(config.get("process", []), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]


class ActionSpaceBuilder:
    """
    Builder for creating ActionSpace from pipeline configuration.

    This class analyzes a pipeline and generates all valid (operator, directive)
    combinations based on directive applicability rules.

    With ModelRegistry, it also generates model-swap actions for LLM operators.

    Example:
        builder = ActionSpaceBuilder(
            directives=[TightenFiltersDirective(), ...],
            model_registry=registry,  # For model-swap actions
        )
        action_space = builder.build(pipeline_config)
    """

    def __init__(
        self,
        directives: Optional[List[Directive]] = None,
        directive_names: Optional[List[str]] = None,
        model_registry: Optional["ModelRegistry"] = None,
        allowed_safety_levels: Optional[List[str]] = None,
    ):
        """
        Initialize the builder.

        Args:
            directives: List of directive instances to use
            directive_names: Names of directives to load from registry
            model_registry: ModelRegistry for generating model-swap actions
            allowed_safety_levels: Safety levels allowed in generated action space
        """
        self._directives: List[Directive] = []
        self._model_registry = model_registry
        self._allowed_safety_levels: Set[str] = set(allowed_safety_levels or ["safe"])

        if directives:
            self._directives.extend(directives)

        if directive_names:
            from agentic_planner.optimizer.directives.registry import DIRECTIVE_REGISTRY

            for name in directive_names:
                if name in DIRECTIVE_REGISTRY:
                    self._directives.append(DIRECTIVE_REGISTRY[name])

    def set_model_registry(self, registry: "ModelRegistry") -> ActionSpaceBuilder:
        """Set the model registry for model-swap actions."""
        self._model_registry = registry
        return self

    def add_directive(self, directive: Directive) -> ActionSpaceBuilder:
        """Add a directive to the builder."""
        self._directives.append(directive)
        return self

    def build(
        self,
        config: DJExecutableConfig,
        include_global: bool = True,
    ) -> ActionSpace:
        """
        Build action space from a pipeline configuration.

        Args:
            config: Pipeline configuration
            include_global: Include global directives (apply to whole pipeline)

        Returns:
            ActionSpace with all valid actions
        """
        process = config.get("process", [])
        index = ProcessIndex.build(process)

        actions: List[Action] = []

        for op_idx, identity in enumerate(index.identities):
            for directive in self._directives:
                if self._is_candidate_allowed(directive, identity, index):
                    action = Action(
                        target_locator=identity.to_target_locator(),
                        operator_name=identity.op_type,
                        directive=directive,
                        operator_identity=identity,
                    )
                    actions.append(action)

            if self._model_registry:
                model_swap_actions = self._build_model_swap_actions(op_idx, identity, config)
                actions.extend(model_swap_actions)

        config_hash = ActionSpace.hash_config(config)

        return ActionSpace(
            actions=actions,
            config_hash=config_hash,
            operator_count=len(index.identities),
        )

    def _build_model_swap_actions(
        self,
        op_idx: int,
        identity: "OpIdentity",
        config: DJExecutableConfig,
    ) -> List[Action]:
        """
        Build model-swap actions for an LLM operator.

        Args:
            op_idx: Operator index
            identity: Operator identity
            config: Pipeline configuration

        Returns:
            List of model-swap actions
        """
        from agentic_planner.optimizer.directives.change_model import SwapSingleOpModelDirective
        from agentic_planner.optimizer.op_locator import OpLocator

        if not self._is_llm_operator(identity):
            return []

        current_model = self._get_current_model(identity.params)
        if not current_model:
            return []

        candidate_models = self._model_registry.get_candidate_models()
        if not candidate_models:
            return []

        actions: List[Action] = []
        for to_model in candidate_models:
            if to_model == current_model:
                continue

            directive = SwapSingleOpModelDirective(
                locator=OpLocator(op_type=identity.op_type, occurrence=op_idx),
                from_model=current_model,
                to_model=to_model,
            )

            action = Action(
                target_locator=identity.to_target_locator(),
                operator_name=identity.op_type,
                directive=directive,
                operator_identity=identity,
            )
            actions.append(action)

        return actions

    def _is_llm_operator(self, identity: "OpIdentity") -> bool:
        """Check if an operator is an LLM-based operator."""
        llm_indicators = [
            "llm_",
            "llm_analysis",
            "llm_filter",
            "llm_map",
            "extract_keyword",
            "extract_entity",
            "extract_event",
            "calibrate_",
            "dialog_",
            "human_preference",
            "optimize_qa",
            "optimize_prompt",
            "generate_qa",
        ]
        op_type = identity.op_type.lower()
        return any(indicator in op_type for indicator in llm_indicators)

    def _get_current_model(self, params: Dict[str, Any]) -> Optional[str]:
        """Get the current model from operator params."""
        model_keys = ["api_model", "api_or_hf_model", "model"]
        for key in model_keys:
            if key in params:
                return params[key]
        return None

    def _is_applicable(
        self,
        directive: Directive,
        identity: OpIdentity,
        index: ProcessIndex,
    ) -> bool:
        """
        Check if a directive is applicable to an operator.

        Args:
            directive: The directive to check
            identity: Operator identity
            index: Process index

        Returns:
            True if the directive can be applied to this operator
        """
        if hasattr(directive, "is_applicable"):
            return directive.is_applicable(identity.op_type, identity.params, index)

        if hasattr(directive, "applicable_op_types"):
            return identity.op_type in directive.applicable_op_types

        return True

    def build_for_operator(
        self,
        config: DJExecutableConfig,
        operator_index: int,
    ) -> List[Action]:
        """
        Build actions for a specific operator.

        Args:
            config: Pipeline configuration
            operator_index: Index of the operator

        Returns:
            List of actions applicable to that operator
        """
        process = config.get("process", [])
        index = ProcessIndex.build(process)

        if operator_index < 0 or operator_index >= len(index.identities):
            return []

        identity = index.identities[operator_index]
        actions: List[Action] = []

        for directive in self._directives:
            if self._is_candidate_allowed(directive, identity, index):
                action = Action(
                    target_locator=identity.to_target_locator(),
                    operator_name=identity.op_type,
                    directive=directive,
                    operator_identity=identity,
                )
                actions.append(action)

        return actions

    def _resolve_directive_spec(self, directive: Directive) -> Optional["DirectiveSpec"]:
        """Resolve directive spec metadata for a directive instance."""
        from agentic_planner.optimizer.directives.registry import get_directive_spec

        spec = get_directive_spec(directive.name)
        if spec is not None:
            return spec

        alias_map = {
            "tighten_filters": "tighten_threshold",
            "loosen_filters": "loosen_threshold",
            "remove_redundant_ops": "remove_redundant_op",
            "reorder_filters_first": "safe_reorder_local",
        }
        spec_name = alias_map.get(directive.name)
        if spec_name is None:
            return None
        return get_directive_spec(spec_name)

    def _is_candidate_allowed(
        self,
        directive: Directive,
        identity: OpIdentity,
        index: ProcessIndex,
    ) -> bool:
        """Check candidate eligibility using explicit spec safety/applicability metadata."""
        spec = self._resolve_directive_spec(directive)
        if spec is not None:
            if not spec.is_search_safe(list(self._allowed_safety_levels)):
                return False

            applicability = spec.applicability
            if not applicability.per_operator:
                return False
            if not applicability.target_locator_supported:
                return False
            if not applicability.allows_operator(identity.op_type):
                return False

        return self._is_applicable(directive, identity, index)


__all__ = [
    "Action",
    "ActionSpace",
    "ActionSpaceBuilder",
    "extract_directive_instantiate_params",
    "resolve_directive_template_name",
]
