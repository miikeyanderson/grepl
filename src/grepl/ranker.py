from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional, Tuple


Source = Literal["grep", "semantic", "hybrid", "ast"]


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
    ast_score: float = 0.0
    ast_pattern: Optional[str] = None
    ast_rule: Optional[str] = None
    ast_captures: Dict[str, str] = field(default_factory=dict)
    # Column-level precision (optional, mainly for AST)
    start_col: Optional[int] = None
    end_col: Optional[int] = None


@dataclass(frozen=True)
class RankWeights:
    semantic: float = 0.45
    grep: float = 0.30
    ast: float = 0.15
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
    ast_score = max(a.ast_score, b.ast_score)

    # Determine source based on which engines contributed
    source: Source
    scores = [
        (ast_score, "ast"),
        (grep_score, "grep"),
        (semantic_score, "semantic"),
    ]
    active = [(s, n) for s, n in scores if s > 0]
    if len(active) >= 2:
        source = "hybrid"
    elif active:
        source = active[0][1]  # type: ignore
    else:
        source = "grep"

    # Prefer AST preview > semantic preview > grep preview
    preview = a.preview
    if b.ast_score > a.ast_score:
        preview = b.preview
    elif b.semantic_score > a.semantic_score and a.ast_score == 0:
        preview = b.preview
    elif a.semantic_score == 0 and b.semantic_score == 0 and a.ast_score == 0:
        preview = b.preview if len(b.preview) > len(a.preview) else a.preview

    symbols = sorted(set((a.symbols or []) + (b.symbols or [])))

    # Merge AST metadata
    ast_pattern = a.ast_pattern or b.ast_pattern
    ast_rule = a.ast_rule or b.ast_rule
    ast_captures = {**a.ast_captures, **b.ast_captures}

    # Merge column info (prefer more precise, i.e., AST)
    start_col = a.start_col if a.ast_score > 0 else b.start_col if b.ast_score > 0 else None
    end_col = a.end_col if a.ast_score > 0 else b.end_col if b.ast_score > 0 else None

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
        ast_score=ast_score,
        ast_pattern=ast_pattern,
        ast_rule=ast_rule,
        ast_captures=ast_captures,
        start_col=start_col,
        end_col=end_col,
    )


def merge_results(
    grep_hits: list[Hit],
    semantic_hits: list[Hit],
    ast_hits: Optional[list[Hit]] = None,
    *,
    overlap_lines: int = 3,
) -> list[Hit]:
    """Merge hits that overlap in the same file, producing hybrid hits when multiple sources agree."""
    merged: list[Hit] = []
    ast_hits = ast_hits or []

    # Group by file for faster overlap checks.
    by_file: Dict[str, list[Hit]] = {}
    for h in [*grep_hits, *semantic_hits, *ast_hits]:
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


def score_hit(
    hit: Hit,
    weights: RankWeights = RankWeights(),
    *,
    ast_exhaustive: bool = False,
) -> float:
    """Score a hit based on source contributions.

    AST scoring philosophy:
    - AST = structural correctness (confirms the match is real)
    - semantic/grep = relevance (how related to the query)
    - AST + semantic/grep = high confidence match
    - AST alone = structurally correct but may not be relevant (lower score in explore mode)
    """
    g = float(max(0.0, min(1.0, hit.grep_score)))
    se = float(max(0.0, min(1.0, hit.semantic_score)))
    ast = float(max(0.0, min(1.0, hit.ast_score)))

    has_grep = g > 0
    has_semantic = se > 0
    has_ast = ast > 0
    has_relevance = has_grep or has_semantic

    # AST + relevance = strong confirmation boost
    if has_ast and has_relevance:
        # AST confirms a relevant match - boost significantly
        base = (weights.semantic * se) + (weights.grep * g)
        # AST confirmation adds a strong boost (not just weighted addition)
        confirmation_boost = 0.20
        s = base + confirmation_boost
        return float(max(0.0, min(1.0, s)))

    # AST only (no relevance signal)
    if has_ast and not has_relevance:
        if ast_exhaustive:
            # In exhaustive/codemod mode, AST-only is valid
            return ast
        else:
            # In explore mode, AST without relevance is lower priority
            # Still include but penalize slightly
            return ast * 0.7

    # Single source (no AST)
    if has_semantic and not has_grep:
        return se
    if has_grep and not has_semantic:
        return g

    # grep + semantic (no AST) - standard hybrid
    if has_grep and has_semantic:
        s = (weights.semantic * se) + (weights.grep * g) + weights.hybrid_boost
        return float(max(0.0, min(1.0, s)))

    return 0.0


def rerank(
    hits: Iterable[Hit],
    *,
    weights: RankWeights = RankWeights(),
    max_per_file: int = 3,
    ast_exhaustive: bool = False,
) -> list[Hit]:
    scored: list[Hit] = []
    for h in hits:
        h.score = score_hit(h, weights, ast_exhaustive=ast_exhaustive)
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
