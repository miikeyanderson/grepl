from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple, TYPE_CHECKING

from .query_expander import get_search_terms
from .diversity import DiversityConfig, mmr_rerank, dedupe_hits
from .ltr import score_with_ltr

if TYPE_CHECKING:
    from .planner import QueryProfile
    from .session import SessionState
    from .user_model import UserProfile


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
    semantic_raw_score: float = 0.0
    semantic_norm_score: float = 0.0
    ast_score: float = 0.0
    ast_pattern: Optional[str] = None
    ast_rule: Optional[str] = None
    ast_captures: Dict[str, str] = field(default_factory=dict)
    # Column-level precision (optional, mainly for AST)
    start_col: Optional[int] = None
    end_col: Optional[int] = None
    language: str = ""
    last_modified: float = 0.0
    symbol_boost: float = 0.0
    lexical_boost: float = 0.0
    recency_boost: float = 0.0
    language_boost: float = 0.0
    user_affinity_boost: float = 0.0
    context_boost: float = 0.0
    graph_boost: float = 0.0


@dataclass(frozen=True)
class RankWeights:
    """Default ranking weights (override via env or CLI for tuning)."""
    semantic: float = 0.5
    grep: float = 0.3
    ast: float = 0.15
    hybrid_boost: float = 0.08
    symbol_boost: float = 0.08
    lexical_boost: float = 0.05
    recency_boost: float = 0.03
    language_boost: float = 0.02
    user_affinity_boost: float = 0.04
    context_boost: float = 0.05
    graph_boost: float = 0.04

    @staticmethod
    def _env_float(name: str) -> Optional[float]:
        value = os.getenv(name)
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @classmethod
    def from_env(cls) -> "RankWeights":
        overrides = {
            "semantic": cls._env_float("GREPL_RANK_SEMANTIC"),
            "grep": cls._env_float("GREPL_RANK_GREP"),
            "ast": cls._env_float("GREPL_RANK_AST"),
            "hybrid_boost": cls._env_float("GREPL_RANK_HYBRID_BOOST"),
            "symbol_boost": cls._env_float("GREPL_RANK_SYMBOL_BOOST"),
            "lexical_boost": cls._env_float("GREPL_RANK_LEXICAL_BOOST"),
            "recency_boost": cls._env_float("GREPL_RANK_RECENCY_BOOST"),
            "language_boost": cls._env_float("GREPL_RANK_LANGUAGE_BOOST"),
            "user_affinity_boost": cls._env_float("GREPL_RANK_USER_AFFINITY_BOOST"),
            "context_boost": cls._env_float("GREPL_RANK_CONTEXT_BOOST"),
            "graph_boost": cls._env_float("GREPL_RANK_GRAPH_BOOST"),
        }
        cleaned = {k: v for k, v in overrides.items() if v is not None}
        return cls(**cleaned) if cleaned else cls()

    def with_overrides(self, **overrides: Optional[float]) -> "RankWeights":
        values = {
            "semantic": self.semantic,
            "grep": self.grep,
            "ast": self.ast,
            "hybrid_boost": self.hybrid_boost,
            "symbol_boost": self.symbol_boost,
            "lexical_boost": self.lexical_boost,
            "recency_boost": self.recency_boost,
            "language_boost": self.language_boost,
            "user_affinity_boost": self.user_affinity_boost,
            "context_boost": self.context_boost,
            "graph_boost": self.graph_boost,
        }
        for key, value in overrides.items():
            if value is not None:
                values[key] = float(value)
        return RankWeights(**values)


class AdaptiveWeights:
    """Adjust ranking weights based on query profile."""

    PRESETS = {
        "identifier": RankWeights(semantic=0.2, grep=0.7, ast=0.3),
        "natural_language": RankWeights(semantic=0.8, grep=0.1, ast=0.15),
        "pattern": RankWeights(semantic=0.1, grep=0.8, ast=0.2),
    }

    @staticmethod
    def blend(base: RankWeights, target: RankWeights, confidence: float) -> RankWeights:
        alpha = max(0.0, min(1.0, confidence))
        return RankWeights(
            semantic=base.semantic * (1 - alpha) + target.semantic * alpha,
            grep=base.grep * (1 - alpha) + target.grep * alpha,
            ast=base.ast * (1 - alpha) + target.ast * alpha,
            hybrid_boost=base.hybrid_boost,
            symbol_boost=base.symbol_boost,
            lexical_boost=base.lexical_boost,
            recency_boost=base.recency_boost,
            language_boost=base.language_boost,
            user_affinity_boost=base.user_affinity_boost,
            context_boost=base.context_boost,
            graph_boost=base.graph_boost,
        )

    @classmethod
    def resolve(cls, profile: "QueryProfile", base: RankWeights) -> RankWeights:
        preset = cls.PRESETS.get(profile.query_type, base)
        return cls.blend(base, preset, profile.confidence)


LANGUAGE_ALIASES = {
    "python": {"py", "python"},
    "typescript": {"ts", "tsx", "typescript"},
    "javascript": {"js", "jsx", "javascript"},
    "tsx": {"tsx", "typescript"},
    "jsx": {"jsx", "javascript"},
    "swift": {"swift"},
    "go": {"go", "golang"},
    "rust": {"rs", "rust"},
    "ruby": {"rb", "ruby"},
    "java": {"java"},
    "kotlin": {"kt", "kotlin"},
    "c": {"c"},
    "cpp": {"cpp", "c++", "cc", "hpp"},
    "markdown": {"md", "markdown"},
    "json": {"json"},
    "yaml": {"yaml", "yml"},
}


def _extract_query_terms(query: str) -> Set[str]:
    """Extract significant terms from a query for symbol matching."""
    stopwords = {"the", "a", "an", "is", "are", "in", "for", "to", "of", "and", "or", "how", "where", "what", "find"}
    words = re.findall(r'\w+', query.lower())
    return {w for w in words if w not in stopwords and len(w) > 2}


def compute_symbol_boost(query: str, symbols: List[str]) -> float:
    """Compute boost based on query term matches in symbols.

    Returns a value between 0.0 and 1.0 indicating how well the symbols match the query.
    """
    if not symbols or not query:
        return 0.0

    query_terms = _extract_query_terms(query)
    if not query_terms:
        return 0.0

    # Check each symbol for matches
    symbol_text = " ".join(symbols).lower()
    matches = 0
    for term in query_terms:
        if term in symbol_text:
            matches += 1

    # Return ratio of matched terms (max 1.0)
    return min(1.0, matches / len(query_terms))


def compute_lexical_overlap(query: str, preview: str, symbols: List[str]) -> float:
    terms = get_search_terms(query)
    if not terms:
        return 0.0
    haystack = f"{preview} {' '.join(symbols)}".lower()
    matches = sum(1 for term in terms if term in haystack)
    return min(1.0, matches / len(terms))


def compute_recency_score(last_modified: float, *, now: float, half_life_days: float = 30.0) -> float:
    if last_modified <= 0:
        return 0.0
    age_days = max(0.0, (now - last_modified) / 86400.0)
    return 0.5 ** (age_days / max(1.0, half_life_days))


def compute_language_match(query: str, language: str) -> float:
    if not query or not language:
        return 0.0
    terms = set(get_search_terms(query))
    lang = language.lower()
    aliases = LANGUAGE_ALIASES.get(lang, {lang})
    return 1.0 if any(alias in terms for alias in aliases) else 0.0


def compute_user_affinity(file_path: str, user_profile: Optional["UserProfile"]) -> float:
    if not user_profile:
        return 0.0
    try:
        return float(user_profile.affinity_for_file(file_path))
    except Exception:
        return 0.0


def _related_file_match(file_path: str, related_imports: Iterable[str]) -> bool:
    if not related_imports:
        return False
    path = Path(file_path)
    stem = path.stem
    for imp in related_imports:
        if not imp:
            continue
        tail = imp.split(".")[-1].split("/")[-1]
        if tail == stem or tail in file_path:
            return True
    return False


def compute_context_match(
    file_path: str,
    session_state: Optional["SessionState"],
    related_imports: Optional[Iterable[str]] = None,
) -> float:
    if not session_state or not session_state.current_file:
        return 0.0
    current = session_state.current_file
    if file_path == current:
        return 1.0
    try:
        if Path(file_path).parent == Path(current).parent:
            return 0.6
    except Exception:
        pass
    if related_imports and _related_file_match(file_path, related_imports):
        return 0.4
    return 0.0


def compute_graph_match(hit: "Hit", graph_files: Optional[Set[str]], graph_symbols: Optional[Set[str]]) -> float:
    if graph_files and hit.file_path in graph_files:
        return 1.0
    if graph_symbols and hit.symbols:
        if set(hit.symbols) & graph_symbols:
            return 1.0
    return 0.0


def _ranges_overlap(a: Tuple[int, int], b: Tuple[int, int], *, within_lines: int = 0) -> bool:
    a0, a1 = a
    b0, b1 = b
    return not (a1 + within_lines < b0 or b1 + within_lines < a0)


def _merge_two(a: Hit, b: Hit) -> Hit:
    start = min(a.start_line, b.start_line)
    end = max(a.end_line, b.end_line)
    grep_score = max(a.grep_score, b.grep_score)
    semantic_score = max(a.semantic_score, b.semantic_score)
    semantic_raw_score = max(a.semantic_raw_score, b.semantic_raw_score)
    semantic_norm_score = max(a.semantic_norm_score, b.semantic_norm_score)
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
    language = a.language or b.language
    last_modified = max(a.last_modified, b.last_modified)

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
        semantic_raw_score=semantic_raw_score,
        semantic_norm_score=semantic_norm_score,
        ast_score=ast_score,
        ast_pattern=ast_pattern,
        ast_rule=ast_rule,
        ast_captures=ast_captures,
        start_col=start_col,
        end_col=end_col,
        language=language,
        last_modified=last_modified,
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
    query: Optional[str] = None,
    now: Optional[float] = None,
    user_profile: Optional["UserProfile"] = None,
    session_state: Optional["SessionState"] = None,
    related_imports: Optional[Iterable[str]] = None,
    graph_files: Optional[Set[str]] = None,
    graph_symbols: Optional[Set[str]] = None,
) -> float:
    """Score a hit based on source contributions.

    AST scoring philosophy:
    - AST = structural correctness (confirms the match is real)
    - semantic/grep = relevance (how related to the query)
    - AST + semantic/grep = high confidence match
    - AST alone = structurally correct but may not be relevant (lower score in explore mode)

    Symbol boost:
    - If query terms appear in chunk symbols (function/class names), add boost
    """
    g = float(max(0.0, min(1.0, hit.grep_score)))
    se = float(max(0.0, min(1.0, hit.semantic_score)))
    ast = float(max(0.0, min(1.0, hit.ast_score)))

    if now is None:
        now = time.time()

    # Compute symbol boost if query provided
    sym_boost = 0.0
    if query and hit.symbols:
        sym_match = compute_symbol_boost(query, hit.symbols)
        sym_boost = weights.symbol_boost * sym_match

    lexical_boost = 0.0
    if query:
        lexical_match = compute_lexical_overlap(query, hit.preview, hit.symbols)
        lexical_boost = weights.lexical_boost * lexical_match

    recency_boost = 0.0
    if hit.last_modified:
        recency_score = compute_recency_score(hit.last_modified, now=now)
        recency_boost = weights.recency_boost * recency_score

    language_boost = 0.0
    if query and hit.language:
        language_match = compute_language_match(query, hit.language)
        language_boost = weights.language_boost * language_match

    user_affinity = compute_user_affinity(hit.file_path, user_profile)
    user_affinity_boost = weights.user_affinity_boost * user_affinity

    context_match = compute_context_match(hit.file_path, session_state, related_imports)
    context_boost = weights.context_boost * context_match

    graph_match = compute_graph_match(hit, graph_files, graph_symbols)
    graph_boost = weights.graph_boost * graph_match

    hit.symbol_boost = sym_boost
    hit.lexical_boost = lexical_boost
    hit.recency_boost = recency_boost
    hit.language_boost = language_boost
    hit.user_affinity_boost = user_affinity_boost
    hit.context_boost = context_boost
    hit.graph_boost = graph_boost

    has_grep = g > 0
    has_semantic = se > 0
    has_ast = ast > 0
    has_relevance = has_grep or has_semantic

    # AST + relevance = strong confirmation boost
    if has_ast and has_relevance:
        base = (weights.semantic * se) + (weights.grep * g)
        confirmation_boost = 0.20
        s = (
            base
            + confirmation_boost
            + sym_boost
            + lexical_boost
            + recency_boost
            + language_boost
            + user_affinity_boost
            + context_boost
            + graph_boost
        )
        return float(max(0.0, min(1.0, s)))

    # AST only (no relevance signal)
    if has_ast and not has_relevance:
        if ast_exhaustive:
            return min(1.0, ast + sym_boost + lexical_boost + recency_boost + language_boost + user_affinity_boost + context_boost + graph_boost)
        else:
            return min(1.0, ast * 0.7 + sym_boost + lexical_boost + recency_boost + language_boost + user_affinity_boost + context_boost + graph_boost)

    # Single source (no AST)
    if has_semantic and not has_grep:
        return min(1.0, se + sym_boost + lexical_boost + recency_boost + language_boost + user_affinity_boost + context_boost + graph_boost)
    if has_grep and not has_semantic:
        return min(1.0, g + sym_boost + lexical_boost + recency_boost + language_boost + user_affinity_boost + context_boost + graph_boost)

    # grep + semantic (no AST) - standard hybrid
    if has_grep and has_semantic:
        s = (
            (weights.semantic * se)
            + (weights.grep * g)
            + weights.hybrid_boost
            + sym_boost
            + lexical_boost
            + recency_boost
            + language_boost
            + user_affinity_boost
            + context_boost
            + graph_boost
        )
        return float(max(0.0, min(1.0, s)))

    return sym_boost + lexical_boost + recency_boost + language_boost + user_affinity_boost + context_boost + graph_boost


def rerank(
    hits: Iterable[Hit],
    *,
    weights: RankWeights = RankWeights(),
    max_per_file: int = 3,
    ast_exhaustive: bool = False,
    query: Optional[str] = None,
    user_profile: Optional["UserProfile"] = None,
    session_state: Optional["SessionState"] = None,
    related_imports: Optional[Iterable[str]] = None,
    graph_files: Optional[Set[str]] = None,
    graph_symbols: Optional[Set[str]] = None,
    ltr_weights: Optional[Dict[str, float]] = None,
    diversity: Optional[DiversityConfig] = None,
) -> list[Hit]:
    scored: list[Hit] = []
    now = time.time()
    for h in hits:
        h.score = score_hit(
            h,
            weights,
            ast_exhaustive=ast_exhaustive,
            query=query,
            now=now,
            user_profile=user_profile,
            session_state=session_state,
            related_imports=related_imports,
            graph_files=graph_files,
            graph_symbols=graph_symbols,
        )
        scored.append(h)

    if ltr_weights:
        blend = 0.5
        for h in scored:
            features = {
                "grep_score": h.grep_score,
                "semantic_score": h.semantic_score,
                "ast_score": h.ast_score,
                "symbol_boost": h.symbol_boost,
                "lexical_boost": h.lexical_boost,
                "recency_boost": h.recency_boost,
                "language_boost": h.language_boost,
                "user_affinity_boost": h.user_affinity_boost,
                "context_boost": h.context_boost,
                "graph_boost": h.graph_boost,
            }
            ltr_score = score_with_ltr(features, ltr_weights)
            h.score = (h.score * (1 - blend)) + (ltr_score * blend)

    scored.sort(key=lambda x: x.score, reverse=True)

    if diversity:
        scored = mmr_rerank(scored, config=diversity, top_k=None)
    else:
        scored = dedupe_hits(scored)

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
