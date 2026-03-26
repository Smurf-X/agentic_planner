# -*- coding: utf-8 -*-
"""Protocol and base utilities for embedding backends."""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    """
    Protocol for embedding backends (local or API).

    All backends must support batch text embedding and single query embedding.
    """

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...

    @property
    def model_name(self) -> str:
        """Return model identifier for caching/logging."""
        ...

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts into vectors.

        :param texts: List of text strings to embed.
        :return: numpy array of shape (len(texts), dimension), dtype float32.
        """
        ...

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        :param query: Query text.
        :return: numpy array of shape (dimension,), dtype float32.
        """
        ...


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize vectors for cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms