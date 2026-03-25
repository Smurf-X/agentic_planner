# -*- coding: utf-8 -*-
"""
Operator locator for stable identification across pipeline transformations.

This module provides mechanisms to locate operators in a process list
without relying on position indices, which become invalid when the
pipeline structure changes.

Key concepts:
- OpIdentity: Stable identity hash for an operator based on its type and params
- OpLocator: Query specification to find an operator
- ProcessIndex: Index structure for efficient lookups
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _stable_hash(obj: Any) -> str:
    """Generate a stable hash for any JSON-serializable object."""
    content = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.md5(content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OpIdentity:
    """
    Immutable identity of an operator in the pipeline.

    The identity is stable: same operator type and params always produce
    the same hash, regardless of position in the process list.
    """

    op_type: str
    """Operator type name, e.g., 'text_length_filter'."""

    params_hash: str
    """Hash of the parameters (excluding metadata fields)."""

    identity_hash: str
    """Unique identifier combining op_type and params_hash."""

    params: Dict[str, Any] = field(compare=False)
    """Original parameters (for matching, not comparison)."""

    @classmethod
    def from_step(cls, step: Dict[str, Any]) -> OpIdentity:
        """
        Create an identity from a process step.

        Args:
            step: A single step from process list, e.g., {"op_name": {"param": value}}

        Returns:
            OpIdentity for this step
        """
        if not isinstance(step, dict) or len(step) != 1:
            raise ValueError(f"Invalid step format: {step}")

        op_type = next(iter(step.keys()))
        raw_params = step.get(op_type, {})
        params = raw_params if isinstance(raw_params, dict) else {}

        # Compute hash excluding any potential metadata fields
        hashable_params = {k: v for k, v in params.items() if not k.startswith("_")}
        params_hash = _stable_hash(hashable_params)[:8]

        # Full identity hash
        identity_content = {"type": op_type, "params": hashable_params}
        identity_hash = _stable_hash(identity_content)[:12]

        return cls(
            op_type=op_type,
            params_hash=params_hash,
            identity_hash=identity_hash,
            params=dict(params),
        )

    def matches_params(self, match_spec: Dict[str, Any]) -> bool:
        """
        Check if params match a specification.

        Args:
            match_spec: Dict of {key: value} to match. Value can be:
                - Exact value to match
                - String starting with "contains:" for substring match

        Returns:
            True if all specified params match
        """
        for key, expected in match_spec.items():
            actual = self.params.get(key)
            if actual is None:
                return False

            if isinstance(expected, str) and expected.startswith("contains:"):
                # Substring match
                substring = expected[9:]  # Remove "contains:" prefix
                if substring not in str(actual):
                    return False
            elif actual != expected:
                return False

        return True


@dataclass
class OpLocator:
    """
    Specification for locating an operator in a process list.

    Supports multiple matching strategies, tried in priority order:
    1. identity_hash: Exact match by identity hash
    2. op_type + param_match: Match by type and parameter values
    3. op_type + occurrence: Match the Nth operator of a type

    Examples:
        # Find specific operator by its identity hash
        OpLocator(identity_hash="a1b2c3d4e5f6")

        # Find llm_filter with specific prompt
        OpLocator(op_type="llm_filter", param_match={"prompt": "contains:summarize"})

        # Find the second text_length_filter
        OpLocator(op_type="text_length_filter", occurrence=1)
    """

    identity_hash: Optional[str] = None
    """Exact match by identity hash."""

    op_type: Optional[str] = None
    """Operator type to match."""

    param_match: Optional[Dict[str, Any]] = None
    """Parameter values to match."""

    occurrence: int = 0
    """Which occurrence of op_type (0-indexed)."""

    def find_index(self, identities: List[OpIdentity]) -> Optional[int]:
        """
        Find the index of the matching operator.

        Args:
            identities: List of operator identities from ProcessIndex

        Returns:
            Index of matching operator, or None if not found
        """
        # Priority 1: Exact hash match
        if self.identity_hash:
            for i, identity in enumerate(identities):
                if identity.identity_hash == self.identity_hash:
                    return i
            return None

        # Priority 2: Type + param match
        if self.op_type and self.param_match:
            for i, identity in enumerate(identities):
                if identity.op_type == self.op_type:
                    if identity.matches_params(self.param_match):
                        return i
            return None

        # Priority 3: Type + occurrence
        if self.op_type:
            count = 0
            for i, identity in enumerate(identities):
                if identity.op_type == self.op_type:
                    if count == self.occurrence:
                        return i
                    count += 1
            return None

        return None

    def find_all(self, identities: List[OpIdentity]) -> List[int]:
        """
        Find all matching operator indices.

        Useful for directives that affect multiple operators.

        Args:
            identities: List of operator identities

        Returns:
            List of matching indices
        """
        results = []

        for i, identity in enumerate(identities):
            # Check identity_hash
            if self.identity_hash:
                if identity.identity_hash == self.identity_hash:
                    results.append(i)
                continue

            # Check op_type
            if self.op_type and identity.op_type != self.op_type:
                continue

            # Check param_match
            if self.param_match and not identity.matches_params(self.param_match):
                continue

            # Check occurrence (only applies when op_type is specified alone)
            if self.op_type and not self.param_match:
                # For find_all, we return all matches regardless of occurrence
                results.append(i)
            else:
                results.append(i)

        return results


@dataclass
class ProcessIndex:
    """
    Index structure for efficient operator lookups.

    Built from a process list and provides:
    - Identity-based lookup
    - Type-based enumeration
    - Locator-based search

    Should be rebuilt whenever the process list changes.
    """

    identities: List[OpIdentity] = field(default_factory=list)
    """List of operator identities, in process order."""

    _hash_to_index: Dict[str, int] = field(default_factory=dict)
    """Mapping from identity_hash to index."""

    _type_to_indices: Dict[str, List[int]] = field(default_factory=dict)
    """Mapping from op_type to list of indices."""

    @classmethod
    def build(cls, process: List[Dict[str, Any]]) -> ProcessIndex:
        """
        Build an index from a process list.

        Args:
            process: The process list from a DJ config

        Returns:
            ProcessIndex ready for lookups
        """
        identities = []
        for step in process:
            try:
                identity = OpIdentity.from_step(step)
                identities.append(identity)
            except ValueError:
                # Skip malformed steps
                continue

        index = cls(identities=identities)

        # Build lookup maps
        for i, identity in enumerate(identities):
            index._hash_to_index[identity.identity_hash] = i

            if identity.op_type not in index._type_to_indices:
                index._type_to_indices[identity.op_type] = []
            index._type_to_indices[identity.op_type].append(i)

        return index

    def locate(self, locator: OpLocator) -> Optional[int]:
        """
        Find an operator using a locator.

        Args:
            locator: The locator specification

        Returns:
            Index of matching operator, or None
        """
        return locator.find_index(self.identities)

    def locate_all(self, locator: OpLocator) -> List[int]:
        """
        Find all operators matching a locator.

        Args:
            locator: The locator specification

        Returns:
            List of matching indices
        """
        return locator.find_all(self.identities)

    def get_by_index(self, index: int) -> Optional[OpIdentity]:
        """Get identity at a specific index."""
        if 0 <= index < len(self.identities):
            return self.identities[index]
        return None

    def get_by_hash(self, hash_: str) -> Optional[Tuple[int, OpIdentity]]:
        """Get operator by identity hash."""
        if hash_ in self._hash_to_index:
            idx = self._hash_to_index[hash_]
            return idx, self.identities[idx]
        return None

    def get_by_type(self, op_type: str) -> List[Tuple[int, OpIdentity]]:
        """Get all operators of a given type."""
        indices = self._type_to_indices.get(op_type, [])
        return [(i, self.identities[i]) for i in indices]

    @property
    def count(self) -> int:
        """Total number of operators."""
        return len(self.identities)

    @property
    def types(self) -> List[str]:
        """List of unique operator types."""
        return list(self._type_to_indices.keys())


__all__ = [
    "OpIdentity",
    "OpLocator",
    "ProcessIndex",
]
