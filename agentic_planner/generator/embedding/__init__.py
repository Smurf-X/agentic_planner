# -*- coding: utf-8 -*-
"""Embedding backends for vector retrieval (local sentence-transformers or API)."""

from __future__ import annotations

from agentic_planner.generator.embedding.backend import EmbeddingBackend, normalize_vectors
from agentic_planner.generator.embedding.local import LocalEmbeddingBackend
from agentic_planner.generator.embedding.api import APIEmbeddingBackend

__all__ = [
    "EmbeddingBackend",
    "normalize_vectors",
    "LocalEmbeddingBackend",
    "APIEmbeddingBackend",
]