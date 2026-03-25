# -*- coding: utf-8 -*-
"""Bump numeric ``min_len`` on ``text_length_filter`` if present."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_planner.contracts.recipe import DJExecutableConfig
from agentic_planner.optimizer.directives.base import Directive, DirectiveResult
from agentic_planner.optimizer.op_locator import ProcessIndex

if TYPE_CHECKING:
    pass


class BumpMinLenDirective(Directive):
    """
    Bump the min_len parameter on text_length_filter operators.

    This is a convenience directive for a common adjustment.
    """

    name = "bump_text_length_min_len"

    def __init__(self, delta: int = 10) -> None:
        """
        Args:
            delta: Amount to increase min_len by
        """
        self.delta = int(delta)

    def apply_with_index(
        self,
        cfg: DJExecutableConfig,
        index: ProcessIndex,
    ) -> DirectiveResult:
        before = self._clone(cfg)
        proc = before.get("process")

        if not isinstance(proc, list):
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="no process",
                config_before=before,
                config_after=before,
            )

        after = self._clone(before)
        applied = False
        affected_hashes = []

        new_proc = []
        for i, step in enumerate(after["process"]):
            op_name, params = self._get_op_params(step)

            if op_name == "text_length_filter":
                cur = params.get("min_len")
                if isinstance(cur, (int, float)):
                    new_params = dict(params)
                    new_params["min_len"] = int(cur) + self.delta
                    new_proc.append({op_name: new_params})
                    applied = True
                    if i < len(index.identities):
                        affected_hashes.append(index.identities[i].identity_hash)
                    continue

            new_proc.append(step)

        after["process"] = new_proc

        return DirectiveResult(
            ok=True,
            applied=applied,
            directive_name=self.name,
            message="bumped min_len" if applied else "no text_length_filter with min_len",
            config_before=before,
            config_after=after,
            details={"delta": self.delta, "affected_identity_hashes": affected_hashes},
        )