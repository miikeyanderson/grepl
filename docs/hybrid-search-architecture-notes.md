# Hybrid Search Architecture Notes

Strategic thinking, engineering principles, and design philosophy for building Cursor-quality hybrid search in grepl.

---

## Why This Matters for Claude Code

Claude Code currently has separate tools that don't talk to each other:
- `Read` - reads files but doesn't know what's relevant
- `Grep` - finds exact text but misses conceptual matches
- `Glob` - finds files by name but not content
- Semantic search exists but is disconnected

**The Problem:** Claude has to guess which tool to use, often picking wrong. It reads entire files when it needs one function. It greps for exact strings when the code was renamed.

**The Solution:** One intelligent command that understands intent and returns exactly what Claude needs, with context.

---

## Design Philosophy

### 1. Intent Over Syntax

Users (and Claude) don't think in tool categories. They think:
- "Where is authentication handled?"
- "Find the handleLogin function"
- "What calls the database?"

The system should understand intent, not require users to pick the right tool.

**Anti-pattern:**
```bash
# User has to know which tool
grepl exact "handleLogin"        # Hope it's exact match
grepl search "login handling"    # Hope semantic works
```

**Target pattern:**
```bash
# System figures it out
grepl find "handleLogin"         # Detects identifier → grep
grepl find "login handling"      # Detects concept → semantic
```

### 2. Precision + Recall Balance

Cursor's genius is balancing two opposing forces:

| Force | Tool | Strength | Weakness |
|-------|------|----------|----------|
| Precision | Grep | Exact, no false positives | Misses renames, synonyms |
| Recall | Semantic | Finds conceptual matches | Can be too broad |

**The merge strategy:**
- Start broad (semantic for recall)
- Tighten with grep (for precision)
- Boost agreement (hybrid confidence)

### 3. Context Is Everything

Finding the line isn't enough. Claude needs:
- The function containing the line
- Related code in the same file
- Connected code in other files
- Enough context to understand, not just locate

**Return chunks, not lines.**

### 4. Speed Is a Feature

If search takes > 500ms, Claude will avoid using it. Cursor feels instant because:
- Embeddings are precomputed
- Vector search is O(log n)
- Grep streams results
- UI shows results as they arrive

**Target: < 200ms for any query.**

---

## Engineering Principles

### Principle 1: Offline Heavy, Online Light

Do expensive work during indexing, not during search.

| Operation | When | Cost |
|-----------|------|------|
| Parse AST | Index time | Expensive |
| Generate embeddings | Index time | Expensive |
| Build vector index | Index time | Expensive |
| Vector lookup | Search time | Cheap |
| Grep search | Search time | Cheap |
| Merge/rerank | Search time | Trivial |

### Principle 2: Graceful Degradation

Never fail completely. Always return something useful.

```
Query: "handleUserAuth"

1. Try grep → No results (function was renamed)
2. Fall back to semantic → Finds "authenticateUser"
3. Return semantic results with note: "No exact match, showing related"
```

```
Query: "authentication middleware flow"

1. Try semantic → Returns 50 vague results
2. Try grep for key terms → Finds "auth", "middleware"
3. Boost semantic results that contain grep hits
4. Return focused hybrid results
```

### Principle 3: Chunk by Meaning, Not Lines

Bad chunking:
```
Chunk 1: lines 1-50
Chunk 2: lines 51-100
Chunk 3: lines 101-150
```

This splits functions in half, losing semantic coherence.

Good chunking:
```
Chunk 1: class AuthService (lines 1-45)
Chunk 2: def authenticate() (lines 47-82)
Chunk 3: def validate_token() (lines 84-120)
```

Each chunk is a complete logical unit.

### Principle 4: Metadata Enriches Everything

Store rich metadata with each chunk:

```python
{
    "symbols": ["AuthService", "authenticate"],
    "imports": ["jwt", "bcrypt"],
    "called_by": ["login_handler", "api_middleware"],
    "calls": ["validate_token", "hash_password"],
    "complexity": 12,
    "last_edited": "2024-01-15",
    "edit_frequency": "high",
    "file_centrality": 0.85,  # How many files import this
}
```

This enables:
- "Find what calls authenticate" → Use `called_by`
- "Find recent auth changes" → Sort by `last_edited`
- "Find core auth code" → Sort by `file_centrality`

### Principle 5: Learn From Claude's Patterns

Track what Claude searches for and what it actually uses:

```python
# Telemetry (local only)
{
    "query": "error handling",
    "results_shown": 10,
    "result_used": "src/errors.py:45-67",
    "time_to_use": 2.3,  # seconds
}
```

Use this to:
- Tune ranking weights
- Identify query patterns
- Improve chunking boundaries
- Add to golden test suite

---

## Query Planning Strategy

### Detection Heuristics

**Strong grep signals:**
- Quoted strings: `"handleLogin"`
- CamelCase: `AuthService`
- snake_case: `user_login`
- Dots: `auth.service`
- Regex metacharacters: `Auth.*Error`
- File extensions: `.py`, `.swift`
- Known identifiers from index

**Strong semantic signals:**
- Question words: "where", "how", "what"
- Action verbs: "find", "show", "get"
- Concept words: "handles", "implements", "validates"
- Descriptions: "the part that...", "code for..."
- Negations: "not the tests", "excluding utils"

**Hybrid signals:**
- Mix of identifier + description
- "Update the AuthService"
- "Find where handleLogin is called"
- Natural language with specific terms

### Confidence Scoring

```python
def plan_query(query: str) -> QueryPlan:
    grep_score = 0.0
    semantic_score = 0.0

    # Check for quoted strings
    if re.search(r'"[^"]+"', query):
        grep_score += 0.5

    # Check for identifiers
    if re.search(r'[A-Z][a-z]+[A-Z]', query):  # CamelCase
        grep_score += 0.3
    if re.search(r'[a-z]+_[a-z]+', query):     # snake_case
        grep_score += 0.3

    # Check for natural language
    if any(w in query.lower() for w in ['where', 'how', 'what', 'find']):
        semantic_score += 0.3
    if len(query.split()) > 3:  # Longer queries tend to be semantic
        semantic_score += 0.2

    # Decide mode
    if grep_score > 0.6 and semantic_score < 0.3:
        return QueryPlan(mode="grep", confidence=grep_score)
    elif semantic_score > 0.6 and grep_score < 0.3:
        return QueryPlan(mode="semantic", confidence=semantic_score)
    else:
        return QueryPlan(mode="hybrid", confidence=max(grep_score, semantic_score))
```

---

## Ranking Strategy

### Weight Tuning

Default weights (tune based on usage data):

```python
WEIGHTS = {
    "semantic_score": 0.40,    # Embedding similarity
    "grep_score": 0.25,        # Exact match presence
    "hybrid_boost": 0.15,      # Bonus if both engines agree
    "symbol_match": 0.10,      # Query terms in symbol names
    "file_importance": 0.05,   # Centrality in codebase
    "recency": 0.05,           # Recently edited files
}
```

### Scoring Formula

```python
def score(hit: Hit, query: str, weights: dict) -> float:
    s = 0.0

    # Base scores
    s += weights["semantic_score"] * hit.semantic_score
    s += weights["grep_score"] * hit.grep_score

    # Hybrid boost (both engines found it)
    if hit.semantic_score > 0.5 and hit.grep_score > 0.5:
        s += weights["hybrid_boost"]

    # Symbol match (query terms appear in function/class names)
    query_terms = set(query.lower().split())
    symbol_terms = set(s.lower() for s in hit.symbols)
    overlap = len(query_terms & symbol_terms) / len(query_terms)
    s += weights["symbol_match"] * overlap

    # File importance (how central is this file)
    s += weights["file_importance"] * hit.file_importance

    # Recency (recently edited = more relevant)
    recency = 1.0 / (1.0 + days_since_edit(hit.last_modified))
    s += weights["recency"] * recency

    return s
```

### Result Diversity

Avoid returning 10 results from the same file. Apply diversity penalty:

```python
def diversify(results: list[Hit], max_per_file: int = 3) -> list[Hit]:
    file_counts = {}
    diverse_results = []

    for hit in results:
        count = file_counts.get(hit.file_path, 0)
        if count < max_per_file:
            diverse_results.append(hit)
            file_counts[hit.file_path] = count + 1

    return diverse_results
```

---

## Chunking Strategy

### Language-Aware Boundaries

Use tree-sitter to find natural boundaries:

**Python:**
- Functions (`def`)
- Classes (`class`)
- Methods (indented `def` under `class`)
- Module-level code blocks

**Swift:**
- Functions (`func`)
- Types (`class`, `struct`, `enum`, `protocol`)
- Extensions (`extension`)
- Computed properties (multi-line)

**TypeScript:**
- Functions (`function`, arrow functions)
- Classes (`class`)
- Interfaces (`interface`)
- React components

### Chunk Size Guidelines

| Size | Pros | Cons |
|------|------|------|
| Too small (< 10 lines) | Precise | Loses context, more chunks to search |
| Too large (> 100 lines) | Complete | Less focused, embedding quality degrades |
| **Optimal (20-60 lines)** | Balanced | Good embedding, good context |

### Overlap Strategy

Include small overlap between adjacent chunks to avoid missing edge cases:

```
Chunk 1: lines 1-50
Chunk 2: lines 45-95   # 5 line overlap
Chunk 3: lines 90-140  # 5 line overlap
```

This ensures nothing falls through the cracks at boundaries.

---

## Embedding Strategy

### Model Selection

| Model | Quality | Speed | Size |
|-------|---------|-------|------|
| OpenAI ada-002 | High | Fast (API) | - |
| Ollama nomic-embed | Good | Fast (local) | 274MB |
| Ollama mxbai-embed | Better | Medium | 669MB |
| CodeBERT | Code-specific | Slow | 440MB |

**Recommendation:** Start with Ollama nomic-embed for local/private. Consider code-specific models for better code understanding.

### Embedding Best Practices

1. **Prepend file context:**
   ```python
   text = f"File: {file_path}\n\n{chunk_content}"
   embedding = embed(text)
   ```

2. **Include symbol names:**
   ```python
   text = f"Symbols: {', '.join(symbols)}\n\n{chunk_content}"
   ```

3. **Normalize code:**
   - Remove excessive whitespace
   - Standardize indentation
   - Strip comments (optional, may lose intent)

4. **Batch for efficiency:**
   ```python
   # Bad: one API call per chunk
   for chunk in chunks:
       embed(chunk)

   # Good: batch API calls
   embeddings = embed_batch(chunks, batch_size=50)
   ```

---

## Performance Optimization

### Indexing Performance

| Technique | Impact |
|-----------|--------|
| Parallel file processing | 4-8x speedup |
| Batch embedding calls | 10x fewer API calls |
| Skip unchanged files | 90% less work on re-index |
| Incremental updates | Near-instant for small changes |

### Search Performance

| Technique | Impact |
|-----------|--------|
| HNSW index | O(log n) vs O(n) for brute force |
| Memory-mapped vectors | Avoid loading entire index |
| Early termination | Stop grep after N hits |
| Parallel engines | Run grep + semantic concurrently |

### Caching Strategy

```python
# Cache embeddings for unchanged files
cache_key = f"{file_path}:{file_hash}"
if cache_key in embedding_cache:
    return embedding_cache[cache_key]

# Cache query embeddings (same query = same vector)
query_cache_key = hash(query)
if query_cache_key in query_embedding_cache:
    query_vec = query_embedding_cache[query_cache_key]
```

---

## Testing Strategy

### Golden Query Suite

Maintain queries with expected results:

```python
GOLDEN_QUERIES = [
    # Exact identifier lookups
    {
        "query": "AuthService",
        "expected_files": ["src/auth/service.py"],
        "expected_mode": "grep",
    },
    # Semantic concept searches
    {
        "query": "where we validate user tokens",
        "expected_files": ["src/auth/tokens.py", "src/middleware/auth.py"],
        "expected_mode": "semantic",
    },
    # Hybrid searches
    {
        "query": "update the AuthService authenticate method",
        "expected_files": ["src/auth/service.py"],
        "expected_mode": "hybrid",
        "expected_symbols": ["authenticate"],
    },
]
```

### Robustness Tests

1. **Rename test:** Rename function, verify semantic still finds it
2. **Synonym test:** Use different words, verify semantic understands
3. **Noise test:** Add irrelevant code, verify ranking stays correct
4. **Scale test:** 100K files, verify latency < 200ms

### A/B Testing Framework

Compare ranking strategies:

```python
def ab_test_ranking(query: str, results_a: list, results_b: list) -> dict:
    """
    Show both result sets, track which one user/Claude actually uses.
    """
    return {
        "query": query,
        "strategy_a": "current",
        "strategy_b": "experimental",
        "winner": track_usage(),
    }
```

---

## Claude Code Integration

### How Claude Should Use It

```
User: "Find where authentication errors are handled"

Claude's thought process:
1. This is a conceptual query → use `grepl find`
2. Run: grepl find "authentication errors are handled"
3. Get hybrid results with context
4. Read the top result for more detail if needed

NOT:
1. Try grepl exact "authentication" → too broad
2. Try grepl exact "auth error" → might miss
3. Try grepl search "auth" → too vague
4. Give up and read all auth files
```

### Optimal Output for Claude

Claude needs:
- **File path** - to reference in responses
- **Line numbers** - for precise citations
- **Code preview** - to understand without re-reading
- **Confidence** - to know if it should look further

```
FIND "authentication error handling" ── 5 results

[hybrid:0.94] src/auth/errors.py:23-45
│ class AuthenticationError(Exception):
│     """Raised when authentication fails."""
│     def __init__(self, reason: str):
│         self.reason = reason
│         super().__init__(f"Auth failed: {reason}")

[semantic:0.82] src/middleware/auth.py:67-89
│ def handle_auth_failure(request, error):
│     """Log and respond to auth failures."""
│     logger.warning(f"Auth failed: {error}")
│     return JSONResponse(status=401, ...)
```

### Error Messages for Claude

When search fails, help Claude recover:

```
FIND "nonexistentFunction" ── 0 results

No exact matches found. Suggestions:
• Try broader terms: grepl find "function"
• Try semantic: grepl find "what does this function do"
• Check spelling: did you mean "existingFunction"?
• List all functions: grepl exact "def " -p src/
```

---

## Competitive Analysis: What Makes Cursor Good

### Things Cursor Does Well

1. **Instant feel** - Results appear as you type
2. **Smart defaults** - Rarely need flags
3. **Context aware** - Knows current file, recent edits
4. **Explains itself** - Shows why results matched
5. **Learns preferences** - Remembers what you clicked

### Things We Can Do Better

1. **CLI-native** - Designed for terminal, not GUI
2. **Transparent** - Show exactly how results were found
3. **Hackable** - Easy to tune weights, add heuristics
4. **Private** - 100% local, no cloud dependency
5. **Claude-optimized** - Output format designed for LLM consumption

---

## Success Criteria

### Functional Requirements

- [ ] Single `grepl find` command handles all search types
- [ ] Query planner correctly detects intent 90%+ of time
- [ ] Hybrid results outperform single-engine results
- [ ] Graceful fallback when one engine fails

### Performance Requirements

- [ ] Search latency < 200ms p95
- [ ] Index 10K files in < 5 minutes
- [ ] Incremental update < 5 seconds
- [ ] Memory usage < 500MB for 100K chunks

### Quality Requirements

- [ ] MRR (Mean Reciprocal Rank) > 0.8
- [ ] Top-1 accuracy > 70%
- [ ] Top-5 accuracy > 90%
- [ ] Claude prefers `grepl find` over separate tools

### User Experience Requirements

- [ ] Zero configuration needed for basic use
- [ ] Helpful error messages with suggestions
- [ ] Progress indication for long operations
- [ ] JSON output for programmatic use

---

## Next Steps

1. **Prototype query planner** - Get intent detection working
2. **Implement basic merge** - Union of grep + semantic
3. **Ship MVP `grepl find`** - Usable but not optimal
4. **Collect usage data** - See what queries Claude runs
5. **Tune ranking weights** - Optimize for real patterns
6. **Add symbol-aware chunking** - Improve embedding quality
7. **Performance optimization** - Hit latency targets
8. **Claude Code integration** - Update hooks to recommend `find`
