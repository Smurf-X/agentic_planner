# -*- coding: utf-8 -*-
"""Search tree node structures with stable operator identities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentic_planner.contracts.recipe import DJExecutableConfig
from agentic_planner.optimizer.op_locator import ProcessIndex, TargetLocator


@dataclass(frozen=True)
class SearchNodeOperator:
    """Operator snapshot tracked within a search node."""

    operator_id: str
    op_type: str
    audit_identity_hash: str
    params: Dict[str, Any]

    def to_target_locator(self) -> TargetLocator:
        """Build canonical target locator for this node operator."""
        return TargetLocator(
            operator_id=self.operator_id,
            audit_identity_hash=self.audit_identity_hash,
        )


@dataclass
class SearchTreeNode:
    """Minimal search node with stable operator identity snapshots."""

    config: DJExecutableConfig
    operators: List[SearchNodeOperator] = field(default_factory=list)
    parent_id: Optional[str] = None
    node_id: str = ""
    depth: int = 0

    @classmethod
    def from_config(
        cls,
        config: DJExecutableConfig,
        parent_id: Optional[str] = None,
        node_id: str = "",
        depth: int = 0,
    ) -> SearchTreeNode:
        """Create a node and capture operator identities from config."""
        node_cfg = deepcopy(config)
        index = ProcessIndex.build(node_cfg.get("process", []))
        operators = [
            SearchNodeOperator(
                operator_id=identity.operator_id,
                op_type=identity.op_type,
                audit_identity_hash=identity.audit_identity_hash,
                params=deepcopy(identity.params),
            )
            for identity in index.identities
        ]
        return cls(
            config=node_cfg,
            operators=operators,
            parent_id=parent_id,
            node_id=node_id,
            depth=depth,
        )


__all__ = ["SearchNodeOperator", "SearchTreeNode"]
