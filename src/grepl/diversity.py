from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from .embedder import get_embeddings


@dataclass(frozen=True)
class DiversityConfig:
    lambda_param: float = 0.7
    min_semantic_distance: float = 0.3
    max_candidates: int = 50


def _content_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def dedupe_hits(hits: Iterable) -> List:
    seen = set()
    out = []
    for hit in hits:
        content = getattr(hit, "preview", "") or ""
        h = _content_hash(content)
        if h in seen:
            continue
        seen.add(h)
        out.append(hit)
    return out


def _cosine_similarity_matrix(vectors: List[List[float]]) -> np.ndarray:
    if not vectors:
        return np.zeros((0, 0), dtype=float)
    mat = np.array(vectors, dtype=float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = mat / norms
    return normalized @ normalized.T


def mmr_rerank(
    hits: List,
    *,
    config: DiversityConfig,
    top_k: Optional[int] = None,
) -> List:
    if not hits:
        return []

    top_k = top_k or len(hits)
    top_k = min(top_k, len(hits))

    candidates = dedupe_hits(hits)[: config.max_candidates]
    if len(candidates) <= 1:
        return candidates

    embeddings = get_embeddings([getattr(h, "preview", "") or "" for h in candidates])
    sim_matrix = _cosine_similarity_matrix(embeddings)

    selected: List[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = float("-inf")
        for idx in remaining:
            relevance = float(getattr(candidates[idx], "score", 0.0))
            if selected:
                max_sim = max(sim_matrix[idx, s] for s in selected)
            else:
                max_sim = 0.0
            distance = 1.0 - max_sim
            if distance < config.min_semantic_distance and selected:
                continue
            mmr_score = (config.lambda_param * relevance) - ((1.0 - config.lambda_param) * max_sim)
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            best_idx = max(remaining, key=lambda i: float(getattr(candidates[i], "score", 0.0)))

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]
