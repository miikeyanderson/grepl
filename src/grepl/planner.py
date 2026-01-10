from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional


Mode = Literal["exact", "semantic", "hybrid"]


@dataclass(frozen=True)
class QueryPlan:
    run_grep: bool
    run_semantic: bool
    grep_pattern: Optional[str]
    grep_fixed: bool
    semantic_query: Optional[str]
    mode: Mode
    confidence: float


# Treat '.' as a literal by default since it's common in identifiers and filenames.
_REGEX_META_RE = re.compile(r"[\\^$|?*+()\[\]{}]")
_QUOTED_RE = re.compile(r"(?:\"([^\"]+)\"|'([^']+)')")
_CAMEL_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b")
_SNAKE_RE = re.compile(r"\b[a-z0-9]+_[a-z0-9_]+\b")
_DOT_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.-]+\b")
_FILE_EXT_RE = re.compile(r"\.[a-zA-Z0-9]{1,6}\b")


def _extract_quoted(query: str) -> list[str]:
    parts: list[str] = []
    for m in _QUOTED_RE.finditer(query):
        parts.append(m.group(1) or m.group(2) or "")
    return [p for p in parts if p]


def _strip_quotes(query: str) -> str:
    return _QUOTED_RE.sub(lambda m: m.group(1) or m.group(2) or "", query)


def _pick_grep_token(query: str) -> Optional[str]:
    quoted = _extract_quoted(query)
    if quoted:
        return quoted[0]

    # Prefer identifier-like tokens
    for rx in (_CAMEL_RE, _SNAKE_RE, _DOT_IDENT_RE):
        m = rx.search(query)
        if m:
            return m.group(0)

    # Fallback: first non-trivial token
    tokens = re.findall(r"[A-Za-z0-9_./:-]+", query)
    tokens = [t for t in tokens if len(t) >= 3]
    if not tokens:
        return None

    stop = {
        "the",
        "and",
        "or",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "from",
        "where",
        "how",
        "what",
        "why",
        "show",
        "find",
        "we",
        "is",
        "are",
        "does",
        "do",
        "it",
        "this",
        "that",
        "a",
        "an",
    }
    filtered = [t for t in tokens if t.lower() not in stop]
    if filtered:
        return max(filtered, key=len)
    return max(tokens, key=len)


def analyze_query(
    query: str,
    *,
    grep_only: bool = False,
    semantic_only: bool = False,
    precise: bool = False,
) -> QueryPlan:
    """Analyze query to decide whether to run grep, semantic, or both."""
    q = query.strip()
    if not q:
        return QueryPlan(
            run_grep=False,
            run_semantic=False,
            grep_pattern=None,
            grep_fixed=False,
            semantic_query=None,
            mode="exact",
            confidence=0.0,
        )

    if grep_only and semantic_only:
        # Prefer explicit grep in conflict.
        semantic_only = False

    quoted = _extract_quoted(q)
    is_just_quoted = bool(_QUOTED_RE.fullmatch(q))
    has_regex_meta = bool(_REGEX_META_RE.search(q))
    has_camel = bool(_CAMEL_RE.search(q))
    has_snake = bool(_SNAKE_RE.search(q))
    has_dot_ident = bool(_DOT_IDENT_RE.search(q))
    has_file_ext = bool(_FILE_EXT_RE.search(q))
    words = re.findall(r"[A-Za-z0-9_]+", q)
    lower_words = [w.lower() for w in words]

    semantic_cues = {
        "where",
        "how",
        "what",
        "why",
        "show",
        "find",
        "handle",
        "handles",
        "implement",
        "implements",
        "validate",
        "validates",
        "called",
        "calls",
        "used",
        "usage",
        "flow",
    }
    has_semantic_words = any(w in semantic_cues for w in lower_words)
    looks_natural = (len(words) >= 4) or has_semantic_words

    # Score both sides with simple heuristics.
    grep_score = 0.0
    semantic_score = 0.0

    if quoted:
        grep_score += 0.7
    if has_camel or has_snake or has_dot_ident:
        grep_score += 0.5
    if has_file_ext:
        grep_score += 0.3
    if has_regex_meta:
        grep_score += 0.6

    if looks_natural:
        semantic_score += 0.6
    if len(words) >= 6:
        semantic_score += 0.2

    if grep_only:
        mode: Mode = "exact"
        run_grep, run_semantic = True, False
    elif semantic_only:
        mode = "semantic"
        run_grep, run_semantic = False, True
    else:
        if is_just_quoted:
            mode = "exact"
            run_grep, run_semantic = True, False
        # Regex-y queries are usually intentional grep.
        elif has_regex_meta and not quoted and not looks_natural:
            mode = "exact"
            run_grep, run_semantic = True, False
        elif grep_score >= 0.9 and semantic_score <= 0.3:
            mode = "exact"
            run_grep, run_semantic = True, False
        elif semantic_score >= 0.7 and grep_score <= 0.4:
            mode = "semantic"
            run_grep, run_semantic = False, True
        else:
            mode = "hybrid"
            run_grep, run_semantic = True, True

    grep_pattern = _pick_grep_token(q) if run_grep else None
    grep_fixed = False
    if run_grep:
        if grep_only:
            # In grep-only mode, treat the full query as the pattern.
            grep_pattern = q
            grep_fixed = not has_regex_meta
        elif grep_pattern and quoted:
            # If user explicitly quoted something, treat it as a literal by default.
            grep_fixed = True

    semantic_query = _strip_quotes(q).strip() if run_semantic else None
    if run_semantic and not semantic_query:
        semantic_query = q

    # Precise mode still needs semantic results, but it will filter at ranking time.
    # (We keep both engines if hybrid; callers can also force grep-only.)
    confidence = max(grep_score, semantic_score)
    if mode == "hybrid":
        confidence = max(0.4, min(0.9, max(grep_score, semantic_score)))
    if precise and run_grep and not run_semantic:
        confidence = max(confidence, 0.8)

    return QueryPlan(
        run_grep=run_grep,
        run_semantic=run_semantic,
        grep_pattern=grep_pattern,
        grep_fixed=grep_fixed,
        semantic_query=semantic_query,
        mode=mode,
        confidence=float(max(0.0, min(1.0, confidence))),
    )
