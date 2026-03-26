# -*- coding: utf-8 -*-
"""BM25 ranking + optional must-include rules over a reduced operator list."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Set, Tuple

from rank_bm25 import BM25Okapi

try:
    import jieba

    _HAS_JIEBA = True
except ImportError:
    jieba = None  # type: ignore[assignment]
    _HAS_JIEBA = False


def _get_operators_registry() -> Dict[str, Any]:
    """Get the operators registry from data_juicer."""
    try:
        from data_juicer.ops.base_op import OPERATORS
        return OPERATORS.modules
    except ImportError:
        return {}


# (alternative keyword set, canonical operator name) — any keyword hit pulls op in.
_MUST_INCLUDE: List[Tuple[Set[str], str]] = [
    ({"抽帧", "均匀", "uniform", "keyframes", "frame_num"}, "video_extract_frames_mapper"),
    ({"时长", "秒", "duration", "min_duration", "过滤短"}, "video_duration_filter"),
    ({"分辨率", "resolution", "resize"}, "video_resolution_filter"),
    ({"场景", "scene", "split"}, "video_split_by_scene_mapper"),
    ({"水印", "watermark"}, "video_watermark_filter"),
    ({"文本长度", "长度", "text_length", "min_len", "length"}, "text_length_filter"),
    ({"去重", "dedup", "minhash", "simhash"}, "document_minhash_deduplicator"),
    ({"caption", "字幕", "captioning"}, "video_captioning_from_frames_mapper"),
    ({"ocr", "文字识别"}, "video_ocr_area_ratio_filter"),
    ({"nsfw", "不安全"}, "video_nsfw_filter"),
]


def _split_operator_name_tokens(name: str) -> List[str]:
    parts = re.split(r"[_\s]+", name.lower())
    return [p for p in parts if len(p) > 1]


def tokenize_query(text: str) -> List[str]:
    """Tokenize user intent for BM25 (Chinese + English)."""
    if not text or not str(text).strip():
        return []
    raw = str(text).strip()
    tokens: List[str] = []
    if _HAS_JIEBA:
        for t in jieba.lcut(raw):
            t = t.strip()
            if len(t) > 1:
                tokens.append(t.lower() if t.isascii() else t)
    else:
        for seg in re.findall(r"[\u4e00-\u9fff]+", raw):
            if len(seg) >= 2:
                tokens.append(seg)
    for m in re.finditer(r"[a-zA-Z][a-zA-Z0-9_]{1,}", raw):
        tokens.append(m.group(0).lower())
    return tokens


def _document_tokens(rec: Any) -> List[str]:
    desc = (rec.desc or "")[:800]
    base = f"{rec.name} {' '.join(rec.tags)} {desc}"
    tks = tokenize_query(base)
    tks.extend(_split_operator_name_tokens(rec.name))
    seen: Set[str] = set()
    out: List[str] = []
    for t in tks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _must_include_names(intent: str, dataset_hint: str) -> List[str]:
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


def rank_candidates(
    intent: str,
    candidates: Sequence,
    *,
    top_k: int = 20,
    dataset_hint: str = "",
) -> List[str]:
    """
    Rank operators with BM25 over ``candidates`` and merge must-include hits.

    Returns up to ``top_k`` operator names (deduplicated, must-include first).
    """
    cands = list(candidates)
    if not cands:
        return []

    name_set = {rec.name for rec in cands}
    must = [n for n in _must_include_names(intent, dataset_hint) if n in name_set]

    if len(cands) == 1:
        out = [cands[0].name]
        for n in must:
            if n not in out:
                out.insert(0, n)
        return out[:top_k]

    corpus = [_document_tokens(rec) for rec in cands]
    q = tokenize_query(f"{intent or ''} {dataset_hint or ''}")
    if not any(corpus):
        names = [rec.name for rec in cands]
        merged = _dedupe_preserve(must + names)
        return merged[:top_k]

    if not q:
        ranked_names = [rec.name for rec in cands]
    else:
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(q)
        order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        ranked_names = [cands[i].name for i in order]

    merged = _dedupe_preserve(must + ranked_names)
    return merged[:top_k]


def _dedupe_preserve(names: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out