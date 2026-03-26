# -*- coding: utf-8 -*-
"""Vector index for operator retrieval with persistence support."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from agentic_planner.generator.embedding.backend import EmbeddingBackend, normalize_vectors

logger = logging.getLogger(__name__)


def _get_op_record_type():
    """Lazy import OPRecord from data_juicer."""
    try:
        from data_juicer.tools.op_search import OPRecord
        return OPRecord
    except ImportError:
        return None


def _make_operator_document(rec: Any) -> str:
    """Build searchable text document for an operator."""
    parts = [
        rec.name,
        " ".join(rec.tags or []),
        (rec.desc or "")[:500],
        " ".join(_split_operator_name(rec.name)),
    ]
    return " ".join(parts)


def _split_operator_name(name: str) -> List[str]:
    """Split operator name into readable tokens."""
    import re
    parts = re.split(r"[_\s]+", name.lower())
    return [p for p in parts if len(p) > 1]


class OperatorVectorIndex:
    """
    Vector index for operator retrieval.

    Features:
    - Build index from OPRecord list
    - Search by query embedding
    - Persist to disk (NPZ format)
    - Cache invalidation by model/operator list hash
    """

    INDEX_VERSION = 1

    def __init__(
        self,
        embedder: EmbeddingBackend,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize vector index.

        :param embedder: Embedding backend for vectorization.
        :param cache_dir: Directory for index cache (default: .embedding_cache).
        """
        self._embedder = embedder
        self._cache_dir = Path(cache_dir or ".embedding_cache")
        self._vectors: Optional[np.ndarray] = None
        self._records: List[Any] = []
        self._name_to_idx: dict[str, int] = {}

    @property
    def is_built(self) -> bool:
        """Check if index is ready for search."""
        return self._vectors is not None and len(self._records) > 0

    def build(self, records: List[Any], show_progress: bool = False) -> None:
        """
        Build vector index from operator records.

        :param records: List of OPRecord objects.
        :param show_progress: Show progress bar during embedding.
        """
        if not records:
            self._vectors = None
            self._records = []
            self._name_to_idx = {}
            return

        logger.info(f"Building vector index for {len(records)} operators...")

        docs = [_make_operator_document(rec) for rec in records]
        self._vectors = self._embedder.embed_texts(docs)

        # L2 normalize for cosine similarity
        if self._vectors is not None and len(self._vectors) > 0:
            self._vectors = normalize_vectors(self._vectors)

        self._records = list(records)
        self._name_to_idx = {rec.name: i for i, rec in enumerate(self._records)}

        logger.info(f"Vector index built: {len(self._records)} operators, dim={self._vectors.shape[1] if self._vectors is not None else 0}")

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Tuple[Any, float]]:
        """
        Search for similar operators by query.

        :param query: Query string.
        :param top_k: Number of results to return.
        :return: List of (OPRecord, score) tuples, sorted by score descending.
        """
        if not self.is_built:
            return []

        q_vec = self._embedder.embed_query(query)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)

        # Cosine similarity (vectors are already normalized)
        scores = np.dot(self._vectors, q_vec)

        # Top-k indices
        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = [
            (self._records[i], float(scores[i]))
            for i in top_indices
            if i < len(self._records)
        ]
        return results

    def save(self, path: Optional[str] = None) -> bool:
        """
        Save index to disk.

        :param path: Custom path (default: cache_dir/model_hash.npz).
        :return: True if saved successfully.
        """
        if not self.is_built:
            return False

        save_path = Path(path) if path else self._get_cache_path()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save vectors
        np.savez_compressed(
            save_path,
            vectors=self._vectors,
            names=[rec.name for rec in self._records],
            version=self.INDEX_VERSION,
            model_name=self._embedder.model_name,
        )

        logger.info(f"Vector index saved to {save_path}")
        return True

    def load(self, records: List[Any], path: Optional[str] = None) -> bool:
        """
        Load index from disk if valid.

        :param records: Current operator list (for validation).
        :param path: Custom path (default: cache_dir/model_hash.npz).
        :return: True if loaded successfully and valid.
        """
        load_path = Path(path) if path else self._get_cache_path()

        if not load_path.exists():
            logger.debug(f"Index cache not found: {load_path}")
            return False

        try:
            data = np.load(load_path, allow_pickle=True)

            # Version check
            if int(data.get("version", 0)) != self.INDEX_VERSION:
                logger.debug(f"Index version mismatch")
                return False

            # Model check
            cached_model = str(data.get("model_name", ""))
            if cached_model != self._embedder.model_name:
                logger.debug(f"Model mismatch: {cached_model} vs {self._embedder.model_name}")
                return False

            # Operator list check
            cached_names = set(data.get("names", []))
            current_names = {rec.name for rec in records}

            if cached_names != current_names:
                logger.debug(f"Operator list changed, rebuild needed")
                return False

            # Load vectors and records
            self._vectors = data["vectors"]
            self._records = list(records)
            self._name_to_idx = {rec.name: i for i, rec in enumerate(self._records)}

            logger.info(f"Vector index loaded from {load_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to load index cache: {e}")
            return False

    def build_or_load(
        self,
        records: List[Any],
        force_rebuild: bool = False,
        show_progress: bool = False,
    ) -> None:
        """
        Load from cache if valid, otherwise build new index.

        :param records: Operator list.
        :param force_rebuild: Force rebuild even if cache exists.
        :param show_progress: Show progress during building.
        """
        if not force_rebuild and self.load(records):
            return

        self.build(records, show_progress=show_progress)
        self.save()

    def _get_cache_path(self) -> Path:
        """Get cache file path based on model name and operator list hash."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Hash operator names for cache invalidation
        names_hash = hashlib.md5(
            json.dumps(sorted(rec.name for rec in self._records), sort_keys=True).encode()
        ).hexdigest()[:8]

        # Sanitize model name for filename
        safe_model_name = self._embedder.model_name.replace("/", "_").replace("\\", "_")
        filename = f"op_index_{safe_model_name}_{names_hash}.npz"

        return self._cache_dir / filename