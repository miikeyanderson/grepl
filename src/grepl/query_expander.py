"""Query expansion for improved semantic code search.

Expands natural language queries into multiple search terms to improve recall.
"""

from typing import List, Set, Optional
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re

from .llm_client import LLMClient

# Common code concept synonyms and related terms
CONCEPT_EXPANSIONS = {
    # Authentication
    "authentication": ["auth", "login", "signin", "sign-in", "authenticate", "session", "token", "jwt", "oauth"],
    "auth": ["authentication", "login", "signin", "authenticate", "session", "token"],
    "login": ["signin", "sign-in", "authenticate", "auth", "session"],
    "logout": ["signout", "sign-out", "session destroy", "clear session"],

    # Database
    "database": ["db", "sql", "query", "model", "repository", "persistence", "storage"],
    "db": ["database", "sql", "query", "model", "repository"],
    "query": ["sql", "database", "fetch", "select", "find"],

    # API/HTTP
    "api": ["endpoint", "route", "handler", "controller", "rest", "http"],
    "endpoint": ["api", "route", "handler", "path"],
    "route": ["endpoint", "path", "url", "handler", "controller"],
    "request": ["http", "api", "fetch", "call"],
    "response": ["http", "return", "result", "reply"],

    # Error handling
    "error": ["exception", "catch", "throw", "failure", "fault", "crash"],
    "exception": ["error", "catch", "throw", "try"],
    "handling": ["handler", "catch", "process", "manage"],

    # Data flow
    "flow": ["process", "pipeline", "sequence", "chain", "workflow"],
    "pipeline": ["flow", "process", "chain", "sequence"],

    # State management
    "state": ["store", "redux", "context", "data", "model"],
    "store": ["state", "redux", "storage", "persist"],

    # Testing
    "test": ["spec", "unit", "integration", "mock", "assert", "expect"],
    "mock": ["stub", "fake", "spy", "test double"],

    # Configuration
    "config": ["configuration", "settings", "options", "preferences", "env"],
    "settings": ["config", "configuration", "options", "preferences"],

    # Validation
    "validation": ["validate", "check", "verify", "sanitize", "parse"],
    "validate": ["check", "verify", "validation", "sanitize"],

    # Cache
    "cache": ["memoize", "store", "persist", "buffer"],

    # Async
    "async": ["await", "promise", "concurrent", "parallel", "task"],
    "concurrent": ["async", "parallel", "thread", "dispatch"],

    # UI
    "render": ["display", "view", "component", "draw", "show"],
    "component": ["view", "widget", "element", "ui"],
    "view": ["screen", "page", "component", "ui", "display"],

    # Navigation
    "navigation": ["navigate", "router", "route", "coordinator", "flow"],
    "router": ["navigation", "route", "coordinator"],

    # Network
    "network": ["http", "api", "fetch", "request", "connection"],
    "fetch": ["request", "get", "load", "download", "http"],

    # File
    "file": ["io", "read", "write", "stream", "path"],
    "read": ["load", "parse", "fetch", "get"],
    "write": ["save", "store", "persist", "output"],

    # Security
    "security": ["auth", "encryption", "permission", "access", "secure"],
    "encryption": ["encrypt", "decrypt", "hash", "crypto"],
    "permission": ["access", "authorize", "role", "acl"],
}

# Words that indicate natural language queries (should use semantic search)
SEMANTIC_CUE_WORDS = {
    "where", "how", "what", "which", "find", "search", "locate",
    "handles", "implements", "does", "manages", "processes",
    "flow", "logic", "related", "responsible", "dealing",
}

LLM_CACHE_PATH = Path.home() / ".grepl" / "llm_cache.json"


@dataclass
class QueryIntent:
    reformulations: List[str]
    code_patterns: List[str]
    concepts: List[str]


def is_natural_language_query(query: str) -> bool:
    """Check if query appears to be natural language (vs code identifier)."""
    words = query.lower().split()

    # Short queries are often identifiers
    if len(words) <= 1:
        return False

    # Check for semantic cue words
    if any(word in SEMANTIC_CUE_WORDS for word in words):
        return True

    # Multi-word queries with common words are likely natural language
    if len(words) >= 3:
        return True

    return False


def _cache_key(query: str) -> str:
    return hashlib.md5(query.encode("utf-8")).hexdigest()


def _load_llm_cache() -> dict:
    if not LLM_CACHE_PATH.exists():
        return {}
    try:
        return json.loads(LLM_CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_llm_cache(cache: dict) -> None:
    LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LLM_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def get_query_intent(query: str, *, llm_expand: bool = False) -> QueryIntent:
    if not llm_expand:
        expansions = expand_query(query, max_expansions=5)
        return QueryIntent(reformulations=expansions, code_patterns=[], concepts=[])

    cache = _load_llm_cache()
    key = _cache_key(query)
    if key in cache:
        cached = cache[key]
        return QueryIntent(
            reformulations=list(cached.get("reformulations", [query])),
            code_patterns=list(cached.get("code_patterns", [])),
            concepts=list(cached.get("concepts", [])),
        )

    client = LLMClient()
    intent = None
    if client.is_available():
        intent = client.expand_query(query)

    if not intent:
        expansions = expand_query(query, max_expansions=5)
        return QueryIntent(reformulations=expansions, code_patterns=[], concepts=[])

    reformulations = intent.get("reformulations") or [query]
    code_patterns = intent.get("code_patterns") or []
    concepts = intent.get("concepts") or []

    payload = {
        "reformulations": reformulations,
        "code_patterns": code_patterns,
        "concepts": concepts,
    }
    cache[key] = payload
    _save_llm_cache(cache)

    return QueryIntent(
        reformulations=list(reformulations),
        code_patterns=list(code_patterns),
        concepts=list(concepts),
    )


def get_expanded_queries(query: str, *, llm_expand: bool = False) -> List[str]:
    intent = get_query_intent(query, llm_expand=llm_expand)
    if not intent.reformulations:
        return [query]
    # Ensure original query is first
    unique = []
    for q in [query, *intent.reformulations]:
        if q not in unique:
            unique.append(q)
    return unique


def expand_query(query: str, max_expansions: int = 5) -> List[str]:
    """Expand a query into multiple related search terms.

    Args:
        query: The original search query
        max_expansions: Maximum number of expanded queries to return

    Returns:
        List of queries including the original and expansions
    """
    queries: Set[str] = {query}
    query_lower = query.lower()

    # Only expand natural language queries
    if not is_natural_language_query(query):
        return [query]

    # Find matching concepts and add their expansions
    words = query_lower.split()
    for word in words:
        if word in CONCEPT_EXPANSIONS:
            expansions = CONCEPT_EXPANSIONS[word]
            for expansion in expansions[:3]:  # Limit expansions per word
                # Create expanded query by replacing the word
                expanded = query_lower.replace(word, expansion)
                queries.add(expanded)

    # Also try adding related terms
    for word in words:
        if word in CONCEPT_EXPANSIONS:
            # Add queries with related terms appended
            for related in CONCEPT_EXPANSIONS[word][:2]:
                queries.add(f"{query_lower} {related}")

    # Preserve original query first, then expansions
    expansions = [q for q in queries if q != query]
    return [query] + expansions[: max(0, max_expansions - 1)]


def get_search_terms(query: str) -> List[str]:
    """Extract key search terms from a query for BM25/keyword matching.

    Returns terms that should be used for keyword-based filtering.
    """
    # Remove common words
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "dare", "ought", "used", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "under",
        "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just",
        "and", "but", "if", "or", "because", "until", "while", "that",
        "which", "who", "whom", "this", "these", "those", "what", "find",
        "search", "locate", "show", "me", "i", "my", "we", "our",
    }

    words = re.findall(r'\w+', query.lower())
    terms = [w for w in words if w not in stopwords and len(w) > 2]

    return terms
