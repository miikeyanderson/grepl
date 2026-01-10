# Hybrid Search Architecture for Grepl

This document outlines the architecture and implementation plan for adding Cursor-style hybrid search to grepl, combining exact text matching (grep) with semantic/meaning-based search.

---

## Overview

### Current State
- `grepl exact` - Text-based pattern matching via ripgrep
- `grepl search` - Semantic search via ChromaDB + Ollama embeddings
- These are separate commands with no integration

### Target State
- Unified `grepl find` command that intelligently combines both approaches
- Query planner that detects intent and dispatches to appropriate engines
- Merge and rerank results for optimal relevance
- Seamless fallback between engines

---

## Core Concepts

### Grep vs Semantic Search

| Aspect | Grep (Exact) | Semantic |
|--------|--------------|----------|
| **Matching** | Literal text, regex | Meaning, concepts |
| **Speed** | Very fast | Fast (precomputed) |
| **Use case** | Known identifiers, error messages | "Find where we handle X" |
| **Weakness** | Misses synonyms, renames | Can be too broad |

### Why Combine Them

1. **Speed**: Semantic is precomputed during indexing; grep adds cheap exactness
2. **Accuracy**: Grep nails literal targets; embeddings surface related code
3. **Fewer misses**: If grep finds nothing (renamed symbols), semantic still works; if semantic is broad, grep tightens it

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Query Planner                          │
│  Analyzes query → detects literals, regex, intent           │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│   Grep Engine   │       │ Semantic Engine │
│   (ripgrep)     │       │ (ChromaDB)      │
└────────┬────────┘       └────────┬────────┘
         │                         │
         └────────────┬────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Merge & Rerank                           │
│  Dedupe → Score → Rank → Return with context                │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Query Planner

**File:** `src/grepl/planner.py`

```python
@dataclass
class QueryPlan:
    run_grep: bool
    run_semantic: bool
    grep_pattern: str | None
    semantic_query: str | None
    mode: Literal["exact", "semantic", "hybrid"]
    confidence: float

def analyze_query(query: str, flags: dict) -> QueryPlan:
    """
    Analyze query to determine search strategy.

    Heuristics:
    - Quoted strings → grep
    - CamelCase/snake_case identifiers → grep
    - Regex patterns → grep only
    - Natural language phrases → semantic
    - Mixed → hybrid
    """
    pass
```

**Detection Rules:**

| Pattern | Strategy | Example |
|---------|----------|---------|
| `"exact phrase"` | Grep only | `"handleUserLogin"` |
| `CamelCase` identifier | Grep first | `AuthService` |
| `snake_case` identifier | Grep first | `user_login` |
| Regex metacharacters | Grep only | `Auth.*Error` |
| Natural language | Semantic first | `where we handle auth` |
| Question format | Semantic | `how does login work` |
| Mixed | Hybrid | `update the handleAuth function` |

### Phase 2: Enhanced Indexing

**Current:** Chunks are created but not optimized for hybrid search.

**Needed:**

1. **Symbol-aware chunking** - Parse AST to chunk by function/class boundaries
2. **Metadata enrichment** - Store symbol names, file importance, edit recency
3. **Incremental updates** - Only re-embed changed chunks

**File:** `src/grepl/indexer.py`

```python
@dataclass
class Chunk:
    id: str
    file_path: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    symbols: list[str]  # function/class names in chunk
    content: str
    embedding: list[float] | None
    last_modified: float

class Indexer:
    def index_file(self, path: Path) -> list[Chunk]:
        """
        1. Parse with tree-sitter for language-aware boundaries
        2. Split into logical chunks (functions, classes, blocks)
        3. Extract symbol names
        4. Generate embeddings
        5. Store in ChromaDB with metadata
        """
        pass

    def incremental_update(self, changed_files: list[Path]):
        """
        1. Delete stale chunks for changed files
        2. Re-chunk and re-embed only changed files
        3. Update metadata (edit recency)
        """
        pass
```

### Phase 3: Merge & Rerank

**File:** `src/grepl/ranker.py`

```python
@dataclass
class Hit:
    source: Literal["grep", "semantic", "hybrid"]
    file_path: str
    start_line: int
    end_line: int
    score: float
    preview: str
    symbols: list[str]

@dataclass
class RankWeights:
    semantic: float = 0.5
    grep: float = 0.3
    proximity: float = 0.1
    recency: float = 0.05
    file_importance: float = 0.05

def merge_results(
    grep_hits: list[Hit],
    semantic_hits: list[Hit],
) -> list[Hit]:
    """
    1. Map grep hits to nearby semantic chunks
    2. Deduplicate overlapping regions
    3. Mark items found by both as "hybrid"
    4. Boost hybrid items (grep + semantic agreement)
    """
    pass

def rerank(
    hits: list[Hit],
    weights: RankWeights,
) -> list[Hit]:
    """
    Final score =
        w1 * semantic_score +
        w2 * grep_score +
        w3 * proximity_to_symbols +
        w4 * edit_recency +
        w5 * file_importance
    """
    pass
```

**Merge Logic:**

1. **Overlap detection**: If grep hit is within N lines of semantic chunk, merge them
2. **Boosting**: Items found by both engines get score boost (1.5x)
3. **Fallback**: If one engine returns nothing, rely on the other
4. **Precision mode**: Require grep confirmation for high-precision queries

### Phase 4: New CLI Command

**File:** `src/grepl/cli.py`

```python
@main.command()
@click.argument("query")
@click.option("-k", "--top-k", default=10, help="Number of results")
@click.option("-p", "--path", default=".", help="Search path")
@click.option("--lang", help="Limit to languages (swift,py,ts)")
@click.option("--grep-only", is_flag=True, help="Only use grep")
@click.option("--semantic-only", is_flag=True, help="Only use semantic")
@click.option("--precise", is_flag=True, help="Require grep confirmation")
@click.option("--json", "json_output", is_flag=True)
def find(query, top_k, path, lang, grep_only, semantic_only, precise, json_output):
    """
    Hybrid search: combines exact matching with semantic understanding.

    Examples:
        grepl find "handleUserLogin"           # Detects identifier → grep
        grepl find "where we handle auth"      # Natural language → semantic
        grepl find "update AuthService logic"  # Mixed → hybrid
    """
    pass
```

**Output Format:**

```
FIND "update auth logic" ── 8 results (hybrid)

[hybrid] src/auth/service.py:45-67 (score: 0.92)
  class AuthService:
      def authenticate(self, user: User) -> bool:
          """Handle user authentication."""
          ...

[semantic] src/middleware/auth.py:12-28 (score: 0.78)
  def auth_middleware(request):
      """Validates auth tokens."""
      ...

[grep] src/tests/test_auth.py:100 (score: 0.71)
  def test_auth_service_login():
```

---

## Data Model

### Chunk Schema (ChromaDB)

```python
{
    "id": "file_path:start_line:end_line",
    "document": "chunk content",
    "embedding": [...],
    "metadata": {
        "file_path": "src/auth/service.py",
        "start_line": 45,
        "end_line": 67,
        "start_byte": 1024,
        "end_byte": 2048,
        "symbols": ["AuthService", "authenticate"],
        "language": "python",
        "last_modified": 1704067200,
        "file_importance": 0.8,  # based on centrality/imports
    }
}
```

### Hit Schema

```python
{
    "source": "hybrid",
    "file_path": "src/auth/service.py",
    "start_line": 45,
    "end_line": 67,
    "score": 0.92,
    "grep_score": 0.85,
    "semantic_score": 0.88,
    "preview": "class AuthService:\n    def authenticate...",
    "symbols": ["AuthService", "authenticate"],
    "tag": "[hybrid]"
}
```

---

## Performance Considerations

### Indexing (Offline)

| Optimization | Implementation |
|--------------|----------------|
| Batch embeddings | Process 10-50 chunks per API call |
| Cache unchanged | Hash file content, skip if unchanged |
| Parallel processing | Use asyncio/multiprocessing per file |
| Incremental sync | Run every 5 min, only process diffs |
| Early availability | Search available at 80% index completion |

### Search (Runtime)

| Optimization | Implementation |
|--------------|----------------|
| Memory-mapped vectors | Use FAISS/HNSWlib for fast top-k |
| Approximate NN | HNSW with tuned recall/speed tradeoff |
| Scoped grep | Limit to specified directories |
| Stream grep results | Don't wait for full completion |
| Result cap | Limit grep to 1000 hits before merge |

### Latency Targets

| Operation | Target |
|-----------|--------|
| Query planning | < 5ms |
| Grep search | < 100ms |
| Semantic search | < 50ms |
| Merge & rerank | < 20ms |
| **Total** | **< 200ms** |

---

## Edge Cases

### No Grep Hits
- Fall back to semantic only
- Expand top-k
- Widen scope if too narrow

### Broad Semantic Matches
- In `--precise` mode, require grep confirmation
- Decay isolated semantic results
- Boost results near grep hits

### Renames/Refactors
- Semantic should still surface relevant code
- Consider storing historical symbol names

### Binary/Large Files
- Skip binary files
- Truncate very large files (> 100KB)
- Exclude vendor/generated directories

---

## File Structure

```
src/grepl/
├── cli.py              # Add 'find' command
├── planner.py          # NEW: Query analysis
├── ranker.py           # NEW: Merge & rerank
├── indexer.py          # ENHANCE: Symbol-aware chunking
├── search.py           # ENHANCE: Integrate with planner
├── models/
│   ├── chunk.py        # NEW: Chunk dataclass
│   └── hit.py          # NEW: Hit dataclass
└── utils/
    └── ast_parser.py   # NEW: Tree-sitter integration
```

---

## CLI Command Summary

### New Command: `grepl find`

```bash
# Auto-detect mode
grepl find "handleUserLogin"              # → grep (identifier)
grepl find "where errors are handled"     # → semantic (natural lang)
grepl find "update the auth middleware"   # → hybrid (mixed)

# Force mode
grepl find "auth" --grep-only
grepl find "authentication flow" --semantic-only

# Options
grepl find "query" -k 20              # Top 20 results
grepl find "query" -p src/            # Search in src/
grepl find "query" --lang python,ts   # Limit languages
grepl find "query" --precise          # Require grep confirmation
grepl find "query" --json             # JSON output
```

### Updated Commands

```bash
# Existing commands remain unchanged
grepl exact "pattern"     # Pure grep (unchanged)
grepl search "query"      # Pure semantic (unchanged)
grepl read file.py        # File reading (unchanged)
grepl context file:line   # Context view (unchanged)

# New unified command
grepl find "query"        # Intelligent hybrid search
```

---

## Testing Strategy

### Golden Queries

Maintain test suite with expected results:

```python
GOLDEN_QUERIES = [
    # Exact matches
    ("handleUserLogin", ["src/auth.py:45"], "grep"),
    ("class AuthService", ["src/auth/service.py:1"], "grep"),

    # Semantic matches
    ("where we validate tokens", ["src/middleware/auth.py"], "semantic"),
    ("error handling logic", ["src/errors.py", "src/handlers.py"], "semantic"),

    # Hybrid matches
    ("update AuthService", ["src/auth/service.py"], "hybrid"),
]
```

### Perturbation Tests

- Rename symbols → verify semantic still finds
- Change formatting → verify grep still matches
- Add synonyms → verify semantic understands

### Performance Tests

- Index 10K files in < 5 minutes
- Search latency < 200ms p95
- Memory usage < 500MB for 100K chunks

---

## Implementation Priority

### MVP (Phase 1)
1. Query planner with basic heuristics
2. Simple merge (union of results)
3. Basic rerank (semantic + grep scores)
4. `grepl find` command with minimal options

### Enhanced (Phase 2)
1. Symbol-aware chunking
2. Overlap-based merge
3. Hybrid boosting
4. Full CLI options

### Advanced (Phase 3)
1. Incremental indexing
2. File importance scoring
3. Edit recency tracking
4. Precision mode

---

## Dependencies

### Current
- `chromadb` - Vector storage
- `ollama` - Local embeddings
- `click` - CLI framework
- `rich` - Output formatting

### New
- `tree-sitter` - AST parsing (already have, enhance usage)
- `tree-sitter-languages` - Language grammars

### Optional
- `faiss-cpu` - Faster vector search (if ChromaDB is slow)
- `watchdog` - File system monitoring for auto-index

---

## Security & Privacy

1. **No remote storage**: Keep all code and embeddings local
2. **Honor ignore files**: Respect .gitignore, .greplignore
3. **Exclude secrets**: Skip .env, credentials, keys
4. **Memory only**: Don't persist plaintext beyond indexing

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Query relevance (MRR) | > 0.8 |
| Search latency p95 | < 200ms |
| Index freshness | < 5 min lag |
| False positive rate | < 10% |
| User preference vs separate commands | > 80% |
