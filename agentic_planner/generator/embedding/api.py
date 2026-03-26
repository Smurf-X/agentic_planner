# -*- coding: utf-8 -*-
"""API-based embedding backend (OpenAI / DashScope compatible)."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

try:
    import httpx

    _HAS_HTTPX = True
except ImportError:
    httpx = None  # type: ignore[misc, assignment]
    _HAS_HTTPX = False


class APIEmbeddingBackend:
    """
    Embedding backend using OpenAI-compatible API.

    Supports:
    - OpenAI: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
    - DashScope (Alibaba Cloud): text-embedding-v3, text-embedding-v2
    - Other OpenAI-compatible endpoints
    """

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSION = 1536
    KNOWN_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "text-embedding-v3": 1024,
        "text-embedding-v2": 1536,
        "text-embedding-v1": 1536,
    }
    BATCH_SIZE = 100

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        dimension: Optional[int] = None,
        timeout_sec: float = 60.0,
    ) -> None:
        """
        Initialize API embedding backend.

        :param api_key: API key (or set OPENAI_API_KEY / DASHSCOPE_API_KEY env var).
        :param base_url: API base URL (or set OPENAI_BASE_URL / DASHSCOPE_BASE_URL env var).
        :param model: Embedding model name.
        :param dimension: Embedding dimension (auto-detected for known models).
        :param timeout_sec: Request timeout.
        """
        if not _HAS_HTTPX:
            raise ImportError(
                "httpx is required for APIEmbeddingBackend. "
                "Install with: pip install httpx"
            )

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        self._base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("DASHSCOPE_BASE_URL") or "").rstrip("/")
        self._model = model
        self._timeout_sec = timeout_sec

        if dimension:
            self._dimension = dimension
        elif model in self.KNOWN_DIMENSIONS:
            self._dimension = self.KNOWN_DIMENSIONS[model]
        else:
            self._dimension = self.DEFAULT_DIMENSION

        if not self._api_key:
            raise ValueError(
                "API key is required. Set api_key parameter or "
                "OPENAI_API_KEY / DASHSCOPE_API_KEY environment variable."
            )
        if not self._base_url:
            raise ValueError(
                "Base URL is required. Set base_url parameter or "
                "OPENAI_BASE_URL / DASHSCOPE_BASE_URL environment variable."
            )

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts via API."""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self._dimension)

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        if not query or not query.strip():
            return np.zeros(self._dimension, dtype=np.float32)
        embeddings = self._embed_batch([query])
        return np.array(embeddings[0], dtype=np.float32)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Call embedding API for a batch of texts."""
        url = f"{self._base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": texts,
            "encoding_format": "float",
        }

        with httpx.Client(timeout=self._timeout_sec) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings