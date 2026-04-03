# -*- coding: utf-8 -*-
"""Operator identity and locator utilities for optimizer actions."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


OPERATOR_ID_PARAM = "_ap_operator_id"


def _stable_hash(obj: Any) -> str:
    """Generate a stable hash for any JSON-serializable object."""
    content = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _new_operator_id() -> str:
    """Generate a unique operator id."""
    return f"op-{uuid.uuid4().hex[:10]}"


@dataclass(frozen=True)
class TargetLocator:
    """Canonical action target reference for a specific operator node."""

    operator_id: str
    audit_identity_hash: str

    def to_dict(self) -> Dict[str, str]:
        """Serialize locator for logs or traces."""
        return {
            "operator_id": self.operator_id,
            "audit_identity_hash": self.audit_identity_hash,
        }

    def canonical_key(self) -> str:
        """Return canonical locator key used in action identity."""
        return f"{self.operator_id}:{self.audit_identity_hash}"


@dataclass(frozen=True)
class OpIdentity:
    """Immutable identity view of a process operator."""

    operator_id: str
    op_type: str
    params_hash: str
    identity_hash: str
    audit_identity_hash: str
    params: Dict[str, Any] = field(compare=False)

    @classmethod
    def from_step(cls, step: Dict[str, Any], fallback_operator_id: Optional[str] = None) -> OpIdentity:
        """Create an identity from a process step."""
        if not isinstance(step, dict) or len(step) != 1:
            raise ValueError(f"Invalid step format: {step}")

        op_type = next(iter(step.keys()))
        raw_params = step.get(op_type, {})
        params = raw_params if isinstance(raw_params, dict) else {}

        operator_id = params.get(OPERATOR_ID_PARAM)
        if not isinstance(operator_id, str) or not operator_id:
            operator_id = fallback_operator_id or _new_operator_id()

        hashable_params = {k: v for k, v in params.items() if k != OPERATOR_ID_PARAM and not k.startswith("_")}
        params_hash = _stable_hash(hashable_params)[:8]

        audit_identity_hash = _stable_hash({"type": op_type, "params": hashable_params})[:12]
        identity_hash = _stable_hash({"id": operator_id, "type": op_type})[:12]

        return cls(
            operator_id=operator_id,
            op_type=op_type,
            params_hash=params_hash,
            identity_hash=identity_hash,
            audit_identity_hash=audit_identity_hash,
            params=dict(params),
        )

    def matches_params(self, match_spec: Dict[str, Any]) -> bool:
        """Check whether params satisfy a matching specification."""
        for key, expected in match_spec.items():
            actual = self.params.get(key)
            if actual is None:
                return False

            if isinstance(expected, str) and expected.startswith("contains:"):
                substring = expected[9:]
                if substring not in str(actual):
                    return False
            elif actual != expected:
                return False

        return True

    def to_target_locator(self) -> TargetLocator:
        """Build the canonical target locator for this identity."""
        return TargetLocator(
            operator_id=self.operator_id,
            audit_identity_hash=self.audit_identity_hash,
        )


@dataclass
class OpLocator:
    """Legacy/utility lookup specification for locating operators."""

    identity_hash: Optional[str] = None
    op_type: Optional[str] = None
    param_match: Optional[Dict[str, Any]] = None
    occurrence: int = 0

    def find_index(self, identities: List[OpIdentity]) -> Optional[int]:
        """Find the first matching index."""
        if self.identity_hash:
            for i, identity in enumerate(identities):
                if identity.identity_hash == self.identity_hash:
                    return i
            return None

        if self.op_type and self.param_match:
            for i, identity in enumerate(identities):
                if identity.op_type == self.op_type and identity.matches_params(self.param_match):
                    return i
            return None

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
        """Find all matching indices."""
        results: List[int] = []

        for i, identity in enumerate(identities):
            if self.identity_hash:
                if identity.identity_hash == self.identity_hash:
                    results.append(i)
                continue

            if self.op_type and identity.op_type != self.op_type:
                continue

            if self.param_match and not identity.matches_params(self.param_match):
                continue

            results.append(i)

        return results


@dataclass
class ProcessIndex:
    """Index structure for stable operator lookup by id and type."""

    identities: List[OpIdentity] = field(default_factory=list)
    _hash_to_index: Dict[str, int] = field(default_factory=dict)
    _id_to_index: Dict[str, int] = field(default_factory=dict)
    _type_to_indices: Dict[str, List[int]] = field(default_factory=dict)

    @classmethod
    def build(cls, process: List[Dict[str, Any]]) -> ProcessIndex:
        """Build an index from process steps, assigning ids if missing."""
        identities: List[OpIdentity] = []

        for step in process:
            try:
                identity = OpIdentity.from_step(step)
            except ValueError:
                continue

            op_params = step.get(identity.op_type)
            if isinstance(op_params, dict):
                op_params.setdefault(OPERATOR_ID_PARAM, identity.operator_id)

            identities.append(identity)

        index = cls(identities=identities)

        for i, identity in enumerate(identities):
            index._hash_to_index[identity.identity_hash] = i
            index._id_to_index[identity.operator_id] = i
            if identity.op_type not in index._type_to_indices:
                index._type_to_indices[identity.op_type] = []
            index._type_to_indices[identity.op_type].append(i)

        return index

    def locate(self, locator: OpLocator) -> Optional[int]:
        """Find an operator using a legacy locator."""
        return locator.find_index(self.identities)

    def locate_all(self, locator: OpLocator) -> List[int]:
        """Find all operators matching a legacy locator."""
        return locator.find_all(self.identities)

    def locate_target(self, target_locator: TargetLocator) -> Optional[int]:
        """Resolve canonical target locator to current index position."""
        idx = self._id_to_index.get(target_locator.operator_id)
        if idx is None:
            return None

        identity = self.identities[idx]
        if identity.audit_identity_hash != target_locator.audit_identity_hash:
            return None
        return idx

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

    def get_by_operator_id(self, operator_id: str) -> Optional[Tuple[int, OpIdentity]]:
        """Get operator by stable operator id."""
        idx = self._id_to_index.get(operator_id)
        if idx is None:
            return None
        return idx, self.identities[idx]

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
    "OPERATOR_ID_PARAM",
    "TargetLocator",
    "OpIdentity",
    "OpLocator",
    "ProcessIndex",
]
