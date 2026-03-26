# -*- coding: utf-8 -*-
"""Move filter-type operators before mapper-type (heuristic)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from agentic_planner.contracts.recipe import DJExecutableConfig
from agentic_planner.optimizer.directives.base import Directive, DirectiveResult
from agentic_planner.optimizer.op_locator import ProcessIndex

if TYPE_CHECKING:
    pass


# Lower runs earlier
_TYPE_PRIORITY: Dict[str, int] = {
    "filter": 0,
    "deduplicator": 1,
    "selector": 2,
    "grouper": 3,
    "mapper": 4,
    "aggregator": 5,
    "pipeline": 6,
    "formatter": 7,
}


_TYPE_CACHE: Dict[str, str] = {}


def _op_type(name: str) -> str:
    """Get the type of an operator by name."""
    if name in _TYPE_CACHE:
        return _TYPE_CACHE[name]

    try:
        from data_juicer.tools.op_search import OPSearcher

        searcher = OPSearcher(specified_op_list=[name])
        if not searcher.op_records:
            t = "mapper"
        else:
            t = searcher.op_records[0].type
    except ImportError:
        t = "mapper"

    _TYPE_CACHE[name] = t
    return t


class ReorderFiltersFirstDirective(Directive):
    """
    Sort process steps by coarse operator type priority (filters first).

    This directive reorders the entire pipeline, moving filter-type
    operators before mapper-type operators to reduce downstream data volume.

    Note: This is a heuristic optimization and does not consider field
    dependencies. Use with caution when filters depend on mapper outputs.

    Note: This is a GLOBAL directive - target_op parameter is ignored.
    """

    name = "reorder_filters_first"
    applicable_op_types = None

    def apply_with_index(
        self,
        cfg: DJExecutableConfig,
        index: ProcessIndex,
        target_op: Optional[int] = None,
    ) -> DirectiveResult:
        before = self._clone(cfg)
        proc = before.get("process")

        if not isinstance(proc, list) or len(proc) < 2:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="nothing to reorder",
                config_before=before,
                config_after=before,
            )

        # Build indexed list with priorities
        indexed: List[tuple[int, int, Dict[str, Any]]] = []
        for i, step in enumerate(proc):
            if not isinstance(step, dict) or len(step) != 1:
                continue
            op_name = next(iter(step.keys()))
            t = _op_type(op_name)
            pri = _TYPE_PRIORITY.get(t, 99)
            indexed.append((pri, i, step))

        if len(indexed) < 2:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="could not classify steps",
                config_before=before,
                config_after=before,
            )

        # Sort by priority (then by original index for stability)
        indexed.sort(key=lambda x: (x[0], x[1]))
        new_proc = [x[2] for x in indexed]
        after = self._clone(before)
        after["process"] = new_proc

        changed = new_proc != proc
        return DirectiveResult(
            ok=True,
            applied=changed,
            directive_name=self.name,
            message="reordered by type priority" if changed else "already ordered",
            config_before=before,
            config_after=after,
            details={
                "priorities": [(x[2], x[0]) for x in indexed],
                "order_before": [
                    index.identities[i].identity_hash for i in range(len(index.identities))
                ],
            },
        )
