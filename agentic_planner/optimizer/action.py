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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from agentic_planner.contracts.recipe import DJExecutableConfig
    from agentic_planner.optimizer.directives.base import Directive, DirectiveResult
    from agentic_planner.optimizer.op_locator import OpIdentity


@dataclass
class Action:
    """
    Unified action abstraction for optimization.

    An action represents applying a directive to a specific operator
    in the pipeline. This is the atomic unit used by search strategies.

    Attributes:
        operator_index: Index of the target operator in process list
        operator_name: Name/type of the target operator
        operator_identity: Identity of the operator (optional, for stable reference)
        directive: The directive to apply
        directive_name: Name of the directive
    """

    operator_index: int
    operator_name: str
    directive: Directive
    operator_identity: Optional[OpIdentity] = None
    directive_name: str = ""

    def __post_init__(self):
        if not self.directive_name:
            self.directive_name = self.directive.name

    def apply(self, config: DJExecutableConfig) -> DirectiveResult:
        """
        Apply the directive to the target operator.

        Args:
            config: Pipeline configuration

        Returns:
            DirectiveResult with outcome
        """
        return self.directive.apply(config, target_op=self.operator_index)

    def __hash__(self) -> int:
        return hash((self.operator_index, self.directive_name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Action):
            return False
        return (
            self.operator_index == other.operator_index
            and self.directive_name == other.directive_name
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize action to dict for logging/debugging."""
        return {
            "operator_index": self.operator_index,
            "operator_name": self.operator_name,
            "directive_name": self.directive_name,
        }

    def __repr__(self) -> str:
        return f"Action({self.operator_name}[{self.operator_index}] -> {self.directive_name})"


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
            used_actions: Set of (operator_index, directive_name) tuples

        Returns:
            New ActionSpace without used actions
        """
        return self.filter(lambda a: (a.operator_index, a.directive_name) not in used_actions)

    def get_for_operator(self, operator_index: int) -> List[Action]:
        """
        Get all actions targeting a specific operator.

        Args:
            operator_index: Index of the operator

        Returns:
            List of actions for that operator
        """
        return [a for a in self.actions if a.operator_index == operator_index]

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

    def group_by_operator(self) -> Dict[int, List[Action]]:
        """
        Group actions by target operator.

        Returns:
            Dict mapping operator_index to list of actions
        """
        groups: Dict[int, List[Action]] = {}
        for action in self.actions:
            if action.operator_index not in groups:
                groups[action.operator_index] = []
            groups[action.operator_index].append(action)
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
            "actions_per_operator": {str(k): len(v) for k, v in by_op.items()},
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

    Example:
        builder = ActionSpaceBuilder(directives=[TightenFiltersDirective(), ...])
        action_space = builder.build(pipeline_config)

        # Use LLM to select top actions
        selected = llm_selector.select(action_space, top_k=10)
    """

    def __init__(
        self,
        directives: Optional[List[Directive]] = None,
        directive_names: Optional[List[str]] = None,
    ):
        """
        Initialize the builder.

        Args:
            directives: List of directive instances to use
            directive_names: Names of directives to load from registry
        """
        self._directives: List[Directive] = []

        if directives:
            self._directives.extend(directives)

        if directive_names:
            from agentic_planner.optimizer.directives.registry import DIRECTIVE_REGISTRY

            for name in directive_names:
                if name in DIRECTIVE_REGISTRY:
                    self._directives.append(DIRECTIVE_REGISTRY[name])

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
        from agentic_planner.optimizer.op_locator import ProcessIndex

        process = config.get("process", [])
        index = ProcessIndex.build(process)

        actions: List[Action] = []

        for op_idx, identity in enumerate(index.identities):
            for directive in self._directives:
                if self._is_applicable(directive, identity, index):
                    action = Action(
                        operator_index=op_idx,
                        operator_name=identity.op_type,
                        directive=directive,
                        operator_identity=identity,
                    )
                    actions.append(action)

        config_hash = ActionSpace.hash_config(config)

        return ActionSpace(
            actions=actions,
            config_hash=config_hash,
            operator_count=len(index.identities),
        )

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
        from agentic_planner.optimizer.op_locator import ProcessIndex

        process = config.get("process", [])
        index = ProcessIndex.build(process)

        if operator_index < 0 or operator_index >= len(index.identities):
            return []

        identity = index.identities[operator_index]
        actions: List[Action] = []

        for directive in self._directives:
            if self._is_applicable(directive, identity, index):
                action = Action(
                    operator_index=operator_index,
                    operator_name=identity.op_type,
                    directive=directive,
                    operator_identity=identity,
                )
                actions.append(action)

        return actions


__all__ = [
    "Action",
    "ActionSpace",
    "ActionSpaceBuilder",
]
