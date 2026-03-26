# -*- coding: utf-8 -*-
"""Concrete instantiated directives with replay signatures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

from agentic_planner.contracts.recipe import DJExecutableConfig
from agentic_planner.optimizer.directives.base import Directive, DirectiveResult
from agentic_planner.optimizer.op_locator import ProcessIndex

if TYPE_CHECKING:
    from agentic_planner.optimizer.directives.specs import DirectiveApplicability
    from agentic_planner.optimizer.op_locator import TargetLocator


@dataclass
class InstantiatedDirective:
    """Runtime directive instance with serializable replay identity."""

    spec_name: str
    directive: Directive
    params: Dict[str, Any] = field(default_factory=dict)
    target_locator: Optional["TargetLocator"] = None
    safety_level: str = "safe"
    applicability: Optional["DirectiveApplicability"] = None
    replay_signature: str = ""

    def __post_init__(self) -> None:
        if not self.replay_signature:
            self.replay_signature = self._compute_replay_signature()

    def _compute_replay_signature(self) -> str:
        """Build a stable replay signature from template and bound arguments."""
        locator_payload: Optional[Dict[str, str]] = None
        if self.target_locator is not None:
            locator_payload = self.target_locator.to_dict()

        payload = {
            "spec_name": self.spec_name,
            "directive_name": self.directive.name,
            "directive_signature": self.directive.replay_signature(),
            "params": self.params,
            "target_locator": locator_payload,
        }
        content = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]

    def apply(self, cfg: DJExecutableConfig) -> DirectiveResult:
        """Apply this instantiated directive against the current config."""
        index = ProcessIndex.build(cfg.get("process", []))

        target_op: Optional[int] = None
        if self.target_locator is not None:
            target_op = index.locate_target(self.target_locator)
            if target_op is None:
                return DirectiveResult(
                    ok=False,
                    applied=False,
                    directive_name=self.spec_name,
                    message=(
                        "target operator no longer exists "
                        f"(operator_id={self.target_locator.operator_id})"
                    ),
                    config_before=cfg,
                    config_after=cfg,
                    details={
                        "invalid_target": True,
                        "target_locator": self.target_locator.to_dict(),
                        "replay_signature": self.replay_signature,
                    },
                )

        return self.directive.apply_with_index(cfg, index, target_op=target_op)
