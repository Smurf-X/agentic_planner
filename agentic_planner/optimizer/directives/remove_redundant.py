# -*- coding: utf-8 -*-
"""Remove redundant or no-op operators from the pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from agentic_planner.contracts.recipe import DJExecutableConfig
from agentic_planner.optimizer.directives.base import Directive, DirectiveResult
from agentic_planner.optimizer.op_locator import ProcessIndex

if TYPE_CHECKING:
    pass


def _is_effective_noop(op_name: str, params: Dict[str, Any]) -> bool:
    """Check if an operator with given params is effectively a no-op."""
    # text_length_filter with no bounds
    if op_name == "text_length_filter":
        min_len = params.get("min_len")
        max_len = params.get("max_len")
        if min_len is None and max_len is None:
            return True
        if min_len == 0 and max_len is None:
            return True

    # language_id_score_filter with any_lang=True
    if op_name == "language_id_score_filter":
        if params.get("any_lang", False):
            return True

    # perplexity_filter with very loose bounds
    if op_name == "perplexity_filter":
        max_ppl = params.get("max_ppl")
        if max_ppl is None or max_ppl >= 1e9:
            return True

    return False


def _find_duplicate_operators(process: List[Dict[str, Any]]) -> Set[int]:
    """Find indices of duplicate operators (same name and params)."""
    seen: Dict[tuple, int] = {}
    duplicates: Set[int] = set()

    for i, step in enumerate(process):
        if not isinstance(step, dict) or len(step) != 1:
            continue
        op_name = next(iter(step.keys()))
        params = step[op_name]
        if not isinstance(params, dict):
            params = {}
        # Create a hashable key from name and sorted params
        key = (op_name, tuple(sorted(params.items())) if params else ())
        if key in seen:
            duplicates.add(i)
        else:
            seen[key] = i

    return duplicates


class RemoveRedundantOpDirective(Directive):
    """
    Remove redundant operators: duplicates, no-ops, and ineffective steps.

    This directive cleans up the pipeline by removing:
    - Duplicate operators (same type and params)
    - No-op operators that don't filter anything

    Note: This is a GLOBAL directive - target_op parameter is ignored.
    """

    name = "remove_redundant_ops"
    applicable_op_types = None

    def __init__(self, remove_duplicates: bool = True, remove_noops: bool = True) -> None:
        """
        Args:
            remove_duplicates: Whether to remove duplicate operators
            remove_noops: Whether to remove no-op operators
        """
        self.remove_duplicates = remove_duplicates
        self.remove_noops = remove_noops

    def apply_with_index(
        self,
        cfg: DJExecutableConfig,
        index: ProcessIndex,
        target_op: Optional[int] = None,
    ) -> DirectiveResult:
        before = self._clone(cfg)
        proc = before.get("process")

        if not isinstance(proc, list) or not proc:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="no process to clean",
                config_before=before,
                config_after=before,
            )

        original_count = len(proc)

        # Find indices to remove
        to_remove: Set[int] = set()

        if self.remove_duplicates:
            to_remove.update(_find_duplicate_operators(proc))

        if self.remove_noops:
            for i, step in enumerate(proc):
                if not isinstance(step, dict) or len(step) != 1:
                    continue
                op_name = next(iter(step.keys()))
                params = step.get(op_name, {})
                if not isinstance(params, dict):
                    params = {}
                if _is_effective_noop(op_name, params):
                    to_remove.add(i)

        if not to_remove:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="no redundant operators found",
                config_before=before,
                config_after=before,
            )

        # Build new process list and track removed identity hashes
        removed_hashes = []
        for i in sorted(to_remove):
            if i < len(index.identities):
                removed_hashes.append(index.identities[i].identity_hash)

        new_proc = [step for i, step in enumerate(proc) if i not in to_remove]
        after = self._clone(before)
        after["process"] = new_proc

        removed_count = original_count - len(new_proc)
        return DirectiveResult(
            ok=True,
            applied=True,
            directive_name=self.name,
            message=f"removed {removed_count} redundant operator(s)",
            config_before=before,
            config_after=after,
            details={
                "removed_count": removed_count,
                "removed_identity_hashes": removed_hashes,
            },
        )
