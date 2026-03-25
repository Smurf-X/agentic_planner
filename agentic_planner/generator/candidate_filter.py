# -*- coding: utf-8 -*-
"""Rule-based narrowing of operator candidates before LLM selection (modality / tags)."""

from __future__ import annotations

import re
from typing import List, Set


def _get_op_record_type():
    """Lazy import OPRecord from data_juicer."""
    try:
        from data_juicer.tools.op_search import OPRecord
        return OPRecord
    except ImportError:
        return None


# Modality tags inferred by OPRecord (see analyze_modality_tag in op_search.py).
MODALITY_TAGS: Set[str] = {"text", "image", "audio", "video"}

# Keywords (Chinese + English) -> modality bucket for coarse intent routing.
MODALITY_KEYWORDS: dict[str, Set[str]] = {
    "video": {
        "video",
        "mp4",
        "avi",
        "mov",
        "mkv",
        "webm",
        "帧",
        "抽帧",
        "关键帧",
        "视频",
        "时长",
        "镜头",
        "scene",
        "clip",
        "frame",
        "keyframes",
        "duration",
        "ffmpeg",
    },
    "image": {
        "image",
        "jpg",
        "jpeg",
        "png",
        "webp",
        "图片",
        "图像",
        "照片",
        "photo",
        "picture",
        "resize",
        "caption",
    },
    "audio": {
        "audio",
        "wav",
        "mp3",
        "flac",
        "语音",
        "音频",
        "speech",
        "asr",
        "sound",
    },
    "text": {
        "text",
        "nlp",
        "句子",
        "段落",
        "文本",
        "token",
        "word",
        "sentence",
        "paragraph",
        "清洗",
        "去重",
        "dedup",
    },
}


def detect_modalities(intent: str, dataset_hint: str) -> Set[str]:
    """
    Return a set of modality tags hinted by user text (may be empty).

    Empty means: do not filter by modality (keep full pool for BM25).
    """
    blob = f"{intent or ''} {dataset_hint or ''}"
    if not blob.strip():
        return set()
    lower = blob.lower()
    found: Set[str] = set()
    for modality, keywords in MODALITY_KEYWORDS.items():
        for kw in keywords:
            if len(kw) <= 1:
                continue
            if re.search(r"[\u4e00-\u9fff]", kw):
                if kw in blob:
                    found.add(modality)
                    break
            else:
                if kw in lower:
                    found.add(modality)
                    break
    return found


def filter_ops_by_modality(records: List, modalities: Set[str]) -> List:
    """
    Keep operators whose tags overlap requested modalities, plus multimodal and
    modality-agnostic (no text/image/audio/video in tags) records.

    If ``modalities`` is empty, return all ``records``.
    """
    if not modalities:
        return list(records)
    out: List = []
    for rec in records:
        tags = set(rec.tags)
        rec_modal = tags & MODALITY_TAGS
        if not rec_modal:
            out.append(rec)
            continue
        if "multimodal" in tags:
            out.append(rec)
            continue
        if tags & modalities:
            out.append(rec)
    return out