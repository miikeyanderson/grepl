# TLDR Features Implementation Plan for Grepl

## Executive Summary

This document outlines a prioritized plan to implement TLDR-style 5-layer architecture features in grepl. The goal is to add call graph analysis, control/data flow analysis, program slicing, improved embeddings, and daemon mode to make grepl more powerful for code understanding and navigation.

---

## Current Grepl Architecture

| Component | Technology | Notes |
|-----------|-----------|-------|
| Embeddings | Ollama + nomic-embed-text | 768 dimensions, cosine similarity |
| Storage | ChromaDB (persistent) | File-based in `~/.grepl/chroma/` |
| AST Search | ast-grep (external tool) | Pattern-based structural search |
| Languages | Multi-language via tree-sitter | Python-specific smart chunking |
| Chunking | Function/class for Python, line-based for others |
| Metadata | file_path, start_line, end_line, symbols, language, last_modified |

---

## TLDR 5-Layer Architecture (Target)

```
Layer 1: AST (Abstract Syntax Tree)        - Structure and syntax
Layer 2: Call Graph                        - Function/method call relationships
Layer 3: CFG (Control Flow Graph)          - Execution paths and branches
Layer 4: DFG (Data Flow Graph)             - Variable dependencies and data flow
Layer 5: PDG (Program Dependence Graph)    - Combined control + data dependencies for slicing
```

---

## Priority Matrix

| Feature | Value | Complexity | Dependencies | Priority |
|---------|-------|------------|--------------|----------|
| Call Graph | High | Medium | None | **P1** |
| Better Embeddings | High | Low | None | **P1** |
| CFG/DFG | Medium | High | Call Graph | P2 |
| Program Slicing | Medium | Very High | CFG + DFG | P3 |
| Daemon Mode | Medium | Medium | All features | P3 |

---

## Phase 1: Call Graph Analyzer (P1 - High Value, Medium Complexity)

### Goal

Enable queries like:
- "Who calls `process_request`?"
- "What functions does `auth_middleware` call?"
- "Show the call chain from `main` to `write_output`"

### Implementation Approach

#### Option A: Python-only (astroid) - Recommended for MVP
**Library**: `astroid` (powers pylint)

**Pros:**
- Already widely used and maintained
- Extended AST with static inference
- Can resolve imports and follow call chains
- Python-only dependency (easy install)

**Cons:**
- Python-only (initially)
- May miss dynamic calls

**Implementation:**
```python
# src/grepl/callgraph.py
from astroid import MANAGER

def build_call_graph(project_path: Path) -> Dict[str, CallGraphNode]:
    """Build call graph for Python project."""
    graph = {}
    for py_file in project_path.rglob("*.py"):
        module = MANAGER.ast_from_file(py_file)
        for node in module.body:
            if isinstance(node, astroid.FunctionDef):
                calls = extract_calls(node)
                graph[node.name] = CallGraphNode(
                    name=node.name,
                    file=py_file,
                    line=node.lineno,
                    calls=calls
                )
    return graph
```

#### Option B: Multi-language (tree-sitter) - Long-term
**Library**: `tree-sitter` (already a dependency)

**Pros:**
- Supports all languages grepl supports
- Consistent API across languages

**Cons:**
- More complex implementation
- Need to build language-specific analyzers
- Less inference capability than astroid

**Design:**
```python
# src/grepl/callgraph/tree_sitter_cg.py
from tree_sitter import Language, Parser

class CallGraphBuilder:
    def __init__(self, language: str):
        self.lang = Language(..., language)
        self.parser = Parser(language=self.lang)

    def extract_calls(self, code: str) -> List[Call]:
        """Extract function calls using tree-sitter queries."""
        query = self.lang.query("""
          (call
            function: (identifier) @func)
        """)
        ...
```

### Data Model

```python
# src/grepl/callgraph/models.py
@dataclass
class CallGraphNode:
    name: str
    file: str
    line: int
    calls: List[str]  # List of function names called
    callers: List[str]  # Inverse: who calls this

@dataclass
class CallChain:
    path: List[CallGraphNode]  # e.g., main -> auth -> validate -> db_query
    depth: int
```

### Storage

Store in ChromaDB as separate collection:
```python
CALL_GRAPH_COLLECTION = "call_graph_{project_fingerprint}"

# Document structure
{
    "id": "module:function_name",
    "metadata": {
        "file": "path/to/file.py",
        "line": 42,
        "calls": ["func1", "func2"],
        "callers": ["main", "process"],
    }
}
```

### CLI Integration

```bash
# New commands
grepl calls <function>              # Show what this function calls
grepl callers <function>            # Show what calls this function
grepl chain <from> <to>             # Show call path between functions

# Output formats
grepl calls process_request --tree   # Tree visualization
grepl calls process_request --dot    # Graphviz DOT output
```

### Effort Estimate
- **Design**: 2-4 hours
- **Implementation (Python-only)**: 8-12 hours
- **Testing**: 4-6 hours
- **Total**: ~14-22 hours

---

## Phase 2: Improved Embeddings (P1 - High Value, Low Complexity)

### Goal

Switch from `nomic-embed-text` to a better model for code search.

### Research Findings

From [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) and [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5):

| Model | Dimensions | MTEB Score | Notes |
|-------|-----------|------------|-------|
| **bge-large-en-v1.5** | 1024 | 64.23% | State-of-the-art, MIT license |
| nomic-embed-text | 768 | ~60% | Current model |
| **voyage-code-2** | 1024 | Unknown | Code-specific, API-only |
| **codebert** | 768 | N/A | Microsoft, older |

### Recommendation: **bge-large-en-v1.5** via Ollama

**Pros:**
- Best general-purpose embeddings (MTEB #1)
- MIT license (commercial use allowed)
- Can run via Ollama: `ollama pull bge-large-en-v1.5` (if available) or use FlagEmbedding library
- 1024 dimensions (better representation)
- Instruction-based retrieval for queries

**Cons:**
- Larger model (0.3B params vs nomic's smaller size)
- May need to switch from Ollama to direct embedding

### Implementation

#### Option A: Ollama (if model available)
```python
# src/grepl/embedder.py
DEFAULT_MODEL = "bge-large-en-v1.5"  # was "nomic-embed-text"
EMBEDDING_DIM = 1024  # was 768

# Migration path
def migrate_embeddings(project_path: Path):
    """Re-index with new model."""
    old_model = "nomic-embed-text"
    new_model = "bge-large-en-v1.5"
    # Re-run indexing with new model
```

#### Option B: FlagEmbedding (direct)
```python
# New dependency
# pip install FlagEmbedding

from FlagEmbedding import FlagModel

class LocalEmbedder:
    def __init__(self):
        self.model = FlagModel(
            'BAAI/bge-large-en-v1.5',
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:"
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)
```

### Effort Estimate
- **Implementation**: 2-4 hours
- **Testing**: 2-3 hours
- **Migration**: 1-2 hours
- **Total**: ~5-9 hours

---

## Phase 3: CFG/DFG Analysis (P2 - Medium Value, High Complexity)

### Goal

Enable queries like:
- "What are all the execution paths through `authenticate`?"
- "What variables flow into this SQL query?"
- "Show the data dependencies for `process_payment`"

### Implementation Approach

#### CFG (Control Flow Graph)

**Library**: Custom using Python's `ast` module

```python
# src/grepl/cfg/builder.py
import ast

class CFGBuilder(ast.NodeVisitor):
    def __init__(self):
        self.blocks = []
        self.edges = []

    def visit_If(self, node):
        # Create conditional branch
        true_block = self.build_block(node.body)
        false_block = self.build_block(node.orelse)
        self.edges.extend([
            (current, true_block, "true"),
            (current, false_block, "false"),
        ])
```

#### DFG (Data Flow Graph)

**Library**: `astroid` for better inference

```python
# src/grepl/dfg/builder.py
class DataFlowAnalyzer:
    def analyze_function(self, func_node):
        """Track variable definitions and uses."""
        definitions = {}
        uses = {}
        for node in func_node.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    definitions[target.id] = node
            # Track uses...
        return definitions, uses
```

### Data Model

```python
@dataclass
class BasicBlock:
    id: str
    statements: List[ast.AST]
    predecessors: List[str]
    successors: List[str]

@dataclass
class DataFlowNode:
    variable: str
    definition_loc: Tuple[str, int]  # (file, line)
    use_locs: List[Tuple[str, int]]
    def_use_chains: List[List[DataFlowNode]]
```

### Effort Estimate
- **CFG Implementation**: 12-16 hours
- **DFG Implementation**: 16-24 hours
- **Testing**: 8-12 hours
- **Total**: ~36-52 hours

---

## Phase 4: Program Slicing (P3 - Medium Value, Very High Complexity)

### Goal

Enable queries like:
- "Show all code that affects this variable"
- "What code is relevant to this line?"

### Implementation Approach

**Library**: Custom PDG (Program Dependence Graph)

The PDG combines:
- **Control Dependence** (from CFG)
- **Data Dependence** (from DFG)

```python
# src/grepl/slicing/pdg.py
class ProgramDependenceGraph:
    def __init__(self, cfg: ControlFlowGraph, dfg: DataFlowGraph):
        self.cfg = cfg
        self.dfg = dfg
        self.nodes = []
        self.edges = []  # (control_dep, data_dep)

    def backward_slice(self, line: int) -> Set[Node]:
        """Compute backward slice from a line."""
        # Find all nodes that affect this line
        worklist = [self.node_at(line)]
        visited = set()
        while worklist:
            node = worklist.pop()
            if node in visited:
                continue
            visited.add(node)
            # Add all predecessors in PDG
            worklist.extend(self.predecessors(node))
        return visited
```

### Effort Estimate
- **PDG Implementation**: 20-30 hours
- **Slice algorithms**: 12-16 hours
- **Testing**: 8-12 hours
- **Total**: ~40-58 hours

---

## Phase 5: Daemon Mode (P3 - Medium Value, Medium Complexity)

### Goal

Run grepl as a background daemon that:
- Maintains in-memory indexes for instant queries
- Watches files for changes and auto-updates
- Provides a faster query interface

### Implementation Approach

#### Architecture

```python
# src/grepl/daemon/server.py
import asyncio
import watchfiles

class GreplDaemon:
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.indexes = {}  # In-memory indexes
        self.running = False

    async def start(self):
        """Start the daemon."""
        # Load initial indexes into memory
        await self.load_indexes()

        # Watch for file changes
        async for changes in watchfiles.awatch(self.project_path):
            await self.handle_changes(changes)

    async def handle_changes(self, changes):
        """Incremental index update."""
        for change, file_path in changes:
            if change == watchfiles.Change.modified:
                await self.update_file(file_path)

    async def query(self, query: str) -> List[Result]:
        """Fast in-memory query."""
        return self.indexes["semantic"].search(query)
```

#### IPC: Unix Socket or HTTP API

```python
# src/grepl/daemon/client.py
import requests

class GreplClient:
    def __init__(self, socket_path: str = "/tmp/grepl.sock"):
        self.base_url = f"http://unix:{socket_path}"

    def search(self, query: str) -> List[Result]:
        resp = requests.post(
            f"{self.base_url}/search",
            json={"query": query}
        )
        return resp.json()
```

### CLI Integration

```bash
# Daemon management
grepl daemon start                 # Start daemon in background
grepl daemon stop                  # Stop daemon
grepl daemon status                # Check if running

# Faster queries when daemon is running
export GREPL_USE_DAEMON=1
grepl search "authentication"      # Uses daemon if available
```

### Effort Estimate
- **Daemon server**: 12-16 hours
- **File watching**: 4-6 hours
- **IPC/client**: 6-8 hours
- **CLI integration**: 4-6 hours
- **Total**: ~26-36 hours

---

## Timeline Summary

| Phase | Feature | Effort | Value | Priority |
|-------|---------|--------|-------|----------|
| 1 | Call Graph (Python) | 14-22h | High | P1 |
| 1 | Better Embeddings | 5-9h | High | P1 |
| 2 | CFG/DFG | 36-52h | Medium | P2 |
| 3 | Program Slicing | 40-58h | Medium | P3 |
| 4 | Daemon Mode | 26-36h | Medium | P3 |

**Total Effort**: ~121-177 hours

**Recommended First Milestone** (P1 only): ~19-31 hours
- Call graph for Python
- Improved embeddings

---

## Implementation Roadmap

### Milestone 1: Foundation (Week 1-2)
1. Implement call graph for Python using astroid
2. Add `grepl calls`, `grepl callers`, `grepl chain` commands
3. Switch to bge-large-en-v1.5 embeddings

### Milestone 2: Flow Analysis (Week 3-5)
1. Implement CFG builder
2. Implement DFG builder
3. Add flow-based queries

### Milestone 3: Advanced Features (Week 6-8)
1. Implement PDG and program slicing
2. Add `grepl slice` command
3. Visualization tools (graph output)

### Milestone 4: Performance (Week 9-10)
1. Implement daemon mode
2. In-memory indexes
3. Auto-updates on file changes

---

## New Dependencies

```toml
[project.dependencies]
"chromadb>=0.4.0",
"click>=8.0",
"rich>=13.0",
"tree-sitter>=0.20.0",
"requests>=2.28.0",
"pygments>=2.14.0",
"astroid>=3.0",           # NEW: Call graph, CFG, DFG
"flagembedding>=0.2.0",   # NEW: Better embeddings
"watchfiles>=0.20",       # NEW: Daemon file watching
"fastapi>=0.100",         # NEW: Daemon HTTP API
"uvicorn>=0.23",          # NEW: Daemon server
```

---

## Open Questions

1. **Multi-language support**: Should call graph support all tree-sitter languages from day 1, or start with Python?
2. **Embedding backend**: Ollama (simpler) or direct FlagEmbedding (faster)?
3. **Daemon communication**: Unix socket or HTTP over localhost?
4. **Backward compatibility**: How to handle existing indexes when upgrading embeddings?

---

## References

- [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) - Improved embeddings
- [astroid](https://github.com/pylint-dev/astroid) - Python code analysis
- [LibCST](https://github.com/Instagram/LibCST) - Concrete Syntax Tree
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/) - Multi-language parsing
- [Program Slicing](https://en.wikipedia.org/wiki/Program_slicing) - Background on slicing
