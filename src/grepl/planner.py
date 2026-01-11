from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


Mode = Literal["exact", "semantic", "hybrid"]
Strategy = Literal["explore", "codemod", "grep"]


@dataclass(frozen=True)
class QueryPlan:
    run_grep: bool
    run_semantic: bool
    run_ast: bool
    grep_pattern: Optional[str]
    grep_fixed: bool
    semantic_query: Optional[str]
    ast_patterns: tuple[str, ...]
    ast_rules: tuple[str, ...]
    ast_language: Optional[str]
    ast_exhaustive: bool
    mode: Mode
    confidence: float

    def describe(self) -> str:
        """Return human-readable description of the execution plan."""
        stages = []
        if self.run_semantic:
            stages.append("semantic")
        if self.run_grep:
            stages.append("grep")
        if self.run_ast:
            mode = "exhaustive" if self.ast_exhaustive else "narrow"
            stages.append(f"ast({mode})")
        return " → ".join(stages) if stages else "none"


@dataclass(frozen=True)
class QueryProfile:
    query_type: Literal["identifier", "natural_language", "pattern"]
    confidence: float
    features: Dict[str, float]


@dataclass
class ExecutionPlan:
    """Rich execution plan with estimated counts and reasoning."""
    query_plan: QueryPlan
    query: str
    path: str

    # Estimated counts (populated during execution)
    semantic_candidate_count: int = 0
    grep_file_count: int = 0
    grep_match_count: int = 0
    ast_file_count: int = 0
    ast_match_count: int = 0

    # Caps for AST stage
    ast_top_files: int = 100
    ast_max_matches: int = 500

    # Reasoning for each stage
    stage_reasons: Dict[str, str] = field(default_factory=dict)

    def add_reason(self, stage: str, reason: str):
        """Add reasoning for why a stage runs."""
        self.stage_reasons[stage] = reason

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "query": self.query,
            "path": self.path,
            "pipeline": self.query_plan.describe(),
            "confidence": self.query_plan.confidence,
            "stages": {
                "semantic": {
                    "enabled": self.query_plan.run_semantic,
                    "query": self.query_plan.semantic_query,
                    "estimated_candidates": self.semantic_candidate_count,
                    "reason": self.stage_reasons.get("semantic", ""),
                } if self.query_plan.run_semantic else None,
                "grep": {
                    "enabled": self.query_plan.run_grep,
                    "pattern": self.query_plan.grep_pattern,
                    "fixed": self.query_plan.grep_fixed,
                    "estimated_files": self.grep_file_count,
                    "estimated_matches": self.grep_match_count,
                    "reason": self.stage_reasons.get("grep", ""),
                } if self.query_plan.run_grep else None,
                "ast": {
                    "enabled": self.query_plan.run_ast,
                    "patterns": list(self.query_plan.ast_patterns),
                    "rules": list(self.query_plan.ast_rules),
                    "language": self.query_plan.ast_language,
                    "exhaustive": self.query_plan.ast_exhaustive,
                    "estimated_files": self.ast_file_count,
                    "estimated_matches": self.ast_match_count,
                    "reason": self.stage_reasons.get("ast", ""),
                } if self.query_plan.run_ast else None,
            },
        }

    def format_human(self) -> str:
        """Format plan for human-readable output with Rich formatting."""
        from rich.console import Console
        from rich.text import Text
        import io

        p = self.query_plan
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, width=100)

        # Header section
        header = Text()
        header.append("Pipeline: ", style="bold cyan")
        header.append(p.describe(), style="white")
        console.print(header)

        line = Text()
        line.append("Query: ", style="dim")
        line.append(self.query or "(none)", style="white")
        console.print(line)

        line = Text()
        line.append("Path: ", style="dim")
        line.append(self.path, style="white")
        console.print(line)

        line = Text()
        line.append("Confidence: ", style="dim")
        line.append(f"{p.confidence:.2f}", style="green")
        console.print(line)
        console.print()

        # Semantic Stage
        if p.run_semantic:
            console.print(Text("Semantic Stage:", style="bold magenta"))
            line = Text("  Query: ", style="dim")
            line.append(str(p.semantic_query), style="white")
            console.print(line)
            if self.semantic_candidate_count:
                line = Text("  Estimated candidates: ", style="dim")
                line.append(str(self.semantic_candidate_count), style="green")
                console.print(line)
            if reason := self.stage_reasons.get("semantic"):
                line = Text("  Why: ", style="dim")
                line.append(reason, style="italic")
                console.print(line)
            console.print()

        # Grep Stage
        if p.run_grep:
            console.print(Text("Grep Stage:", style="bold yellow"))
            line = Text("  Pattern: ", style="dim")
            line.append(str(p.grep_pattern), style="white")
            console.print(line)
            line = Text("  Fixed string: ", style="dim")
            line.append(str(p.grep_fixed), style="white")
            console.print(line)
            if self.grep_file_count:
                line = Text("  Estimated files: ", style="dim")
                line.append(str(self.grep_file_count), style="green")
                console.print(line)
            if self.grep_match_count:
                line = Text("  Estimated matches: ", style="dim")
                line.append(str(self.grep_match_count), style="green")
                console.print(line)
            if reason := self.stage_reasons.get("grep"):
                line = Text("  Why: ", style="dim")
                line.append(reason, style="italic")
                console.print(line)
            console.print()

        # AST Stage
        if p.run_ast:
            console.print(Text("AST Stage:", style="bold blue"))
            if p.ast_patterns:
                line = Text("  Patterns: ", style="dim")
                line.append(str(list(p.ast_patterns)), style="white")
                console.print(line)
            if p.ast_rules:
                line = Text("  Rules: ", style="dim")
                line.append(str(list(p.ast_rules)), style="white")
                console.print(line)
            if p.ast_language:
                line = Text("  Language: ", style="dim")
                line.append(p.ast_language, style="white")
                console.print(line)
            line = Text("  Mode: ", style="dim")
            mode_text = "exhaustive (full repo)" if p.ast_exhaustive else "narrow (filtered files)"
            line.append(mode_text, style="white")
            console.print(line)
            if not p.ast_exhaustive:
                line = Text("  File cap: ", style="dim")
                line.append(str(self.ast_top_files), style="green")
                console.print(line)
                line = Text("  Match cap: ", style="dim")
                line.append(str(self.ast_max_matches), style="green")
                console.print(line)
            if self.ast_file_count:
                line = Text("  Estimated files to scan: ", style="dim")
                line.append(str(self.ast_file_count), style="green")
                console.print(line)
            if reason := self.stage_reasons.get("ast"):
                line = Text("  Why: ", style="dim")
                line.append(reason, style="italic")
                console.print(line)

        return buffer.getvalue().rstrip()


# Treat '.' as a literal by default since it's common in identifiers and filenames.
_REGEX_META_RE = re.compile(r"[\\^$|?*+()\[\]{}]")
_QUOTED_RE = re.compile(r"(?:\"([^\"]+)\"|'([^']+)')")
_CAMEL_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*[A-Z][A-Za-z0-9]*\b")
_SNAKE_RE = re.compile(r"\b[a-z0-9]+_[a-z0-9_]+\b")
_DOT_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.-]+\b")
_FILE_EXT_RE = re.compile(r"\.[a-zA-Z0-9]{1,6}\b")


def is_identifier_like_query(query: str) -> bool:
    """Heuristic to detect identifier-like or short technical queries."""
    q = query.strip()
    if not q:
        return False

    if _CAMEL_RE.search(q) or _SNAKE_RE.search(q) or _DOT_IDENT_RE.search(q):
        return True
    if _FILE_EXT_RE.search(q):
        return True

    words = re.findall(r"[A-Za-z0-9_]+", q)
    if len(words) <= 2 and any(len(w) >= 3 for w in words):
        return True

    return False


def profile_query(query: str) -> QueryProfile:
    q = query.strip()
    words = re.findall(r"[A-Za-z0-9_]+", q)
    word_count = len(words)
    if word_count == 0:
        return QueryProfile(query_type="identifier", confidence=0.5, features={"word_count": 0})

    camel = len([w for w in words if _CAMEL_RE.search(w)])
    camel_ratio = camel / max(1, word_count)

    stop = {
        "the", "a", "an", "is", "are", "in", "for", "to", "of", "and", "or", "how", "where", "what",
        "find", "show", "with", "from", "does", "do", "using", "use",
    }
    stop_count = len([w for w in words if w.lower() in stop])
    stop_ratio = stop_count / max(1, word_count)

    has_regex = 1.0 if _REGEX_META_RE.search(q) else 0.0

    identifier_score = 0.0
    natural_score = 0.0
    pattern_score = 0.0

    if is_identifier_like_query(q):
        identifier_score += 0.6
    if camel_ratio >= 0.4:
        identifier_score += 0.3

    if word_count >= 4 or stop_ratio >= 0.3:
        natural_score += 0.6
    if word_count >= 6:
        natural_score += 0.2

    if has_regex:
        pattern_score += 0.7
    if re.search(r"[\\^$|?*+()\\[\\]{}]", q):
        pattern_score += 0.2

    scores = {
        "identifier": identifier_score,
        "natural_language": natural_score,
        "pattern": pattern_score,
    }
    query_type = max(scores, key=scores.get)
    confidence = max(0.2, min(1.0, scores[query_type]))

    return QueryProfile(
        query_type=query_type,  # type: ignore
        confidence=confidence,
        features={
            "word_count": float(word_count),
            "camel_ratio": float(camel_ratio),
            "stop_ratio": float(stop_ratio),
            "regex": float(has_regex),
        },
    )


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
    ast_patterns: Optional[List[str]] = None,
    ast_rules: Optional[List[str]] = None,
    ast_language: Optional[str] = None,
    ast_exhaustive: bool = False,
    strategy: Optional[Strategy] = None,
) -> QueryPlan:
    """Analyze query to decide whether to run grep, semantic, ast, or combinations."""
    q = query.strip()
    ast_patterns = ast_patterns or []
    ast_rules = ast_rules or []
    run_ast = bool(ast_patterns or ast_rules)

    if not q and not run_ast:
        return QueryPlan(
            run_grep=False,
            run_semantic=False,
            run_ast=False,
            grep_pattern=None,
            grep_fixed=False,
            semantic_query=None,
            ast_patterns=tuple(ast_patterns),
            ast_rules=tuple(ast_rules),
            ast_language=ast_language,
            ast_exhaustive=ast_exhaustive,
            mode="exact",
            confidence=0.0,
        )

    # Handle AST-only with no query: don't run semantic/grep, just AST
    if not q and run_ast:
        # If no query but AST patterns/rules provided, run AST-only
        # Use exhaustive if specified, otherwise still AST-only (no narrowing possible)
        return QueryPlan(
            run_grep=False,
            run_semantic=False,
            run_ast=True,
            grep_pattern=None,
            grep_fixed=False,
            semantic_query=None,
            ast_patterns=tuple(ast_patterns),
            ast_rules=tuple(ast_rules),
            ast_language=ast_language,
            ast_exhaustive=ast_exhaustive or True,  # Force exhaustive when no narrowing possible
            mode="exact",
            confidence=0.8 if ast_exhaustive else 0.5,
        )

    # Handle strategy presets
    if strategy == "codemod":
        # AST-only exhaustive search
        return QueryPlan(
            run_grep=False,
            run_semantic=False,
            run_ast=True,
            grep_pattern=None,
            grep_fixed=False,
            semantic_query=None,
            ast_patterns=tuple(ast_patterns),
            ast_rules=tuple(ast_rules),
            ast_language=ast_language,
            ast_exhaustive=True,
            mode="exact",
            confidence=0.9,
        )
    elif strategy == "grep":
        # Grep-only
        grep_only = True

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
        run_ast=run_ast,
        grep_pattern=grep_pattern,
        grep_fixed=grep_fixed,
        semantic_query=semantic_query,
        ast_patterns=tuple(ast_patterns),
        ast_rules=tuple(ast_rules),
        ast_language=ast_language,
        ast_exhaustive=ast_exhaustive,
        mode=mode,
        confidence=float(max(0.0, min(1.0, confidence))),
    )
