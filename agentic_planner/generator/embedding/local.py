# -*- coding: utf-8 -*-
"""Local embedding backend using sentence-transformers."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST = True
except ImportError:
    SentenceTransformer = None  # type: ignore[misc, assignment]
    _HAS_ST = False


class LocalEmbeddingBackend:
    """
    Local embedding backend using sentence-transformers.

    Recommended models:
    - BAAI/bge-base-zh-v1.5: ~400MB, good for Chinese (default)
    - BAAI/bge-small-zh-v1.5: ~100MB, faster but less accurate
    - BAAI/bge-m3: ~2GB, multilingual with large context
    """

    DEFAULT_MODEL = "BAAI/bge-base-zh-v1.5"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
    ) -> None:
        """
        Initialize local embedding model.

        :param model_name: HuggingFace model identifier.
        :param device: "cuda", "cpu", or None (auto-detect).
        :param cache_folder: Custom cache directory for model weights.
        """
        if not _HAS_ST:
            raise ImportError(
                "sentence-transformers is required for LocalEmbeddingBackend. "
                "Install with: pip install sentence-transformers"
            )
        self._model_name = model_name
        self._model = SentenceTransformer(
            model_name_or_path=model_name,
            device=device,
            cache_folder=cache_folder,
        )
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts into vectors."""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        if not query or not query.strip():
            return np.zeros(self._dimension, dtype=np.float32)
        embedding = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)