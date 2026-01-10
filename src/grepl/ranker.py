from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Tuple


Source = Literal["grep", "semantic", "hybrid"]


@dataclass
class Hit:
    source: Source
    file_path: str
    start_line: int
    end_line: int
    score: float
    preview: str
    symbols: list[str]
    grep_score: float = 0.0
    semantic_score: float = 0.0


@dataclass(frozen=True)
class RankWeights:
    semantic: float = 0.55
    grep: float = 0.35
    hybrid_boost: float = 0.10


def _ranges_overlap(a: Tuple[int, int], b: Tuple[int, int], *, within_lines: int = 0) -> bool:
    a0, a1 = a
    b0, b1 = b
    return not (a1 + within_lines < b0 or b1 + within_lines < a0)


def _merge_two(a: Hit, b: Hit) -> Hit:
    start = min(a.start_line, b.start_line)
    end = max(a.end_line, b.end_line)
    grep_score = max(a.grep_score, b.grep_score)
    semantic_score = max(a.semantic_score, b.semantic_score)
    source: Source
    if grep_score > 0 and semantic_score > 0:
        source = "hybrid"
    else:
        source = "grep" if grep_score > 0 else "semantic"

    # Prefer semantic preview if available, otherwise grep.
    preview = a.preview
    if b.semantic_score > a.semantic_score:
        preview = b.preview
    elif a.semantic_score == 0 and b.semantic_score == 0:
        # Both grep: keep the larger preview (more context).
        preview = b.preview if len(b.preview) > len(a.preview) else a.preview

    symbols = sorted(set((a.symbols or []) + (b.symbols or [])))
    return Hit(
        source=source,
        file_path=a.file_path,
        start_line=start,
        end_line=end,
        score=0.0,
        preview=preview,
        symbols=symbols,
        grep_score=grep_score,
        semantic_score=semantic_score,
    )


def merge_results(
    grep_hits: list[Hit],
    semantic_hits: list[Hit],
    *,
    overlap_lines: int = 3,
) -> list[Hit]:
    """Merge hits that overlap in the same file, producing hybrid hits when both agree."""
    merged: list[Hit] = []

    # Group by file for faster overlap checks.
    by_file: Dict[str, list[Hit]] = {}
    for h in [*grep_hits, *semantic_hits]:
        by_file.setdefault(h.file_path, []).append(h)

    for _, hits in by_file.items():
        hits = sorted(hits, key=lambda x: (x.start_line, x.end_line))
        bucket: list[Hit] = []
        for h in hits:
            placed = False
            for i, existing in enumerate(bucket):
                if _ranges_overlap(
                    (existing.start_line, existing.end_line),
                    (h.start_line, h.end_line),
                    within_lines=overlap_lines,
                ):
                    bucket[i] = _merge_two(existing, h)
                    placed = True
                    break
            if not placed:
                bucket.append(h)
        merged.extend(bucket)

    return merged


def score_hit(hit: Hit, weights: RankWeights = RankWeights()) -> float:
    g = float(max(0.0, min(1.0, hit.grep_score)))
    se = float(max(0.0, min(1.0, hit.semantic_score)))

    # If only one engine contributed, keep that engine's score scale.
    if g > 0 and se == 0:
        return g
    if se > 0 and g == 0:
        return se

    s = (weights.semantic * se) + (weights.grep * g)
    if g > 0 and se > 0:
        s += weights.hybrid_boost
    return float(max(0.0, min(1.0, s)))


def rerank(
    hits: Iterable[Hit],
    *,
    weights: RankWeights = RankWeights(),
    max_per_file: int = 3,
) -> list[Hit]:
    scored: list[Hit] = []
    for h in hits:
        h.score = score_hit(h, weights)
        scored.append(h)

    scored.sort(key=lambda x: x.score, reverse=True)

    if max_per_file <= 0:
        return scored

    # Simple diversity cap.
    counts: Dict[str, int] = {}
    out: list[Hit] = []
    for h in scored:
        c = counts.get(h.file_path, 0)
        if c >= max_per_file:
            continue
        counts[h.file_path] = c + 1
        out.append(h)
    return out
