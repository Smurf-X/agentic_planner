# -*- coding: utf-8 -*-
"""Tests for stable operator ids and locator targeting."""

from __future__ import annotations

from copy import deepcopy

from agentic_planner.optimizer.op_locator import ProcessIndex
from agentic_planner.optimizer.search.tree_node import SearchTreeNode


def _sample_process():
    return [
        {"text_length_filter": {"min_len": 10, "max_len": 1000}},
        {"words_num_filter": {"min_num": 5, "max_num": 100}},
        {"perplexity_filter": {"max_ppl": 500}},
    ]


def test_process_index_preserves_operator_ids_across_reorder_and_param_edits():
    process = _sample_process()
    index = ProcessIndex.build(process)

    root_ids = [identity.operator_id for identity in index.identities]
    assert len(set(root_ids)) == 3

    reordered_process = [process[2], process[0], process[1]]
    reordered_index = ProcessIndex.build(reordered_process)
    reordered_ids = [identity.operator_id for identity in reordered_index.identities]

    assert set(reordered_ids) == set(root_ids)

    step_params = reordered_process[0]["perplexity_filter"]
    step_params["max_ppl"] = 450
    edited_index = ProcessIndex.build(reordered_process)

    assert edited_index.identities[0].operator_id == reordered_ids[0]


def test_search_tree_node_builds_target_locators_from_stable_ids():
    config = {"process": deepcopy(_sample_process())}
    node = SearchTreeNode.from_config(config)

    assert len(node.operators) == 3
    assert all(op.operator_id for op in node.operators)
    assert all(op.to_target_locator().operator_id == op.operator_id for op in node.operators)
