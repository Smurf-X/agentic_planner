# -*- coding: utf-8 -*-
"""Vector-based candidate retriever for operator selection."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from agentic_planner.generator.candidate_filter import detect_modalities, filter_ops_by_modality
from agentic_planner.generator.candidate_ranker import _MUST_INCLUDE
from agentic_planner.generator.embedding.backend import EmbeddingBackend
from agentic_planner.generator.vector_index import OperatorVectorIndex

logger = logging.getLogger(__name__)


def _get_operators_registry() -> Dict[str, Any]:
    """Get the operators registry from data_juicer."""
    try:
        from data_juicer.ops.base_op import OPERATORS
        return OPERATORS.modules
    except ImportError:
        return {}


def _must_include_names(intent: str, dataset_hint: str) -> List[str]:
    """Extract must-include operator names from intent (reuse from candidate_ranker)."""
    blob = f"{intent or ''} {dataset_hint or ''}"
    if not blob:
        return []
    lower = blob.lower()
    names: List[str] = []
    operators = _get_operators_registry()
    for kws, op_name in _MUST_INCLUDE:
        if op_name not in operators:
            continue
        hit = False
        for kw in kws:
            if re.search(r"[\u4e00-\u9fff]", kw):
                if kw in blob:
                    hit = True
                    break
            else:
                if kw.lower() in lower:
                    hit = True
                    break
        if hit:
            names.append(op_name)
    return names


class VectorRetriever:
    """
    Vector-based retriever for operator candidates.

    Uses embedding similarity to find relevant operators, with optional
    modality filtering and must-include rules.
    """

    def __init__(
        self,
        embedder: EmbeddingBackend,
        cache_dir: str = ".embedding_cache",
        include_formatter: bool = False,
    ) -> None:
        """
        Initialize vector retriever.

        :param embedder: Embedding backend (local or API).
        :param cache_dir: Directory for index cache.
        :param include_formatter: Include formatter operators.
        """
        self._embedder = embedder
        self._cache_dir = cache_dir
        self._include_formatter = include_formatter
        self._index: Optional[OperatorVectorIndex] = None
        self._all_records: List[Any] = []

    def _ensure_index(self) -> None:
        """Build or load index if not ready."""
        if self._index is not None and self._index.is_built:
            return

        try:
            from data_juicer.tools.op_search import OPSearcher
        except ImportError:
            logger.warning("data_juicer not installed, vector retrieval unavailable")
            return

        searcher = OPSearcher(include_formatter=self._include_formatter)
        self._all_records = list(searcher.op_records)

        self._index = OperatorVectorIndex(
            embedder=self._embedder,
            cache_dir=self._cache_dir,
        )
        self._index.build_or_load(self._all_records, force_rebuild=False)

    def retrieve(
        self,
        intent: str,
        top_k: int = 20,
        dataset_hint: str = "",
        use_modality_filter: bool = True,
        use_must_include: bool = True,
    ) -> List[str]:
        """
        Retrieve relevant operator names by vector similarity.

        :param intent: User's natural language intent.
        :param top_k: Maximum number of operators to return.
        :param dataset_hint: Additional context from dataset.
        :param use_modality_filter: Pre-filter by detected modality.
        :param use_must_include: Apply must-include rules.
        :return: List of operator names, must-include first.
        """
        self._ensure_index()

        if not self._index or not self._index.is_built:
            logger.warning("Index not ready, returning empty list")
            return []

        # Modality filtering (optional pre-filter)
        if use_modality_filter:
            modalities = detect_modalities(intent, dataset_hint)
            if modalities:
                filtered_records = filter_ops_by_modality(self._all_records, modalities)
                if filtered_records:
                    # Build temporary index for filtered set
                    query = f"{intent or ''} {dataset_hint or ''}"
                    results = self._index.search(query, top_k=min(top_k * 2, 50))

                    # Filter results to match modality-filtered set
                    allowed_names = {rec.name for rec in filtered_records}
                    filtered_results = [
                        (rec, score) for rec, score in results
                        if rec.name in allowed_names
                    ][:top_k]
                else:
                    filtered_results = []
            else:
                query = f"{intent or ''} {dataset_hint or ''}"
                filtered_results = self._index.search(query, top_k=top_k)
        else:
            query = f"{intent or ''} {dataset_hint or ''}"
            filtered_results = self._index.search(query, top_k=top_k)

        # Extract names
        names = [rec.name for rec, score in filtered_results]

        # Must-include rules
        if use_must_include:
            must_names = _must_include_names(intent, dataset_hint)
            names = self._merge_must_include(names, must_names)

        return names[:top_k]

    def _merge_must_include(self, ranked_names: List[str], must_names: List[str]) -> List[str]:
        """Merge must-include names at the front, preserving order."""
        if not must_names:
            return ranked_names

        # Filter must_names to those that exist in operator registry
        valid_must = [n for n in must_names if n in {rec.name for rec in self._all_records}]

        # Deduplicate: remove must names from ranked list
        remaining = [n for n in ranked_names if n not in set(valid_must)]

        return valid_must + remaining

    def rebuild_index(self, force: bool = True) -> None:
        """
        Force rebuild the vector index.

        :param force: Force rebuild even if cache exists.
        """
        try:
            from data_juicer.tools.op_search import OPSearcher
        except ImportError:
            logger.warning("data_juicer not installed, cannot rebuild index")
            return

        searcher = OPSearcher(include_formatter=self._include_formatter)
        self._all_records = list(searcher.op_records)

        self._index = OperatorVectorIndex(
            embedder=self._embedder,
            cache_dir=self._cache_dir,
        )
        self._index.build_or_load(self._all_records, force_rebuild=force, show_progress=True)