# Grepl Feature Roadmap

Track requested features, their status, and implementation details.

---

## Completed

### Tree-Style Output
**Status:** Done (v0.2.5)

Replaced Rich panels with compact tree-style output optimized for Claude Code and terminal workflows.

```
EXACT "def " ── 8 matches in 2 files

├── src/grepl/cli.py (5)
│   ├── 39: def main():
│   ├── 47: def index(path, force):
│   └── ...
└── src/grepl/store.py (3)
    ├── 27: def get_client():
    └── ...
```

**Features:**
- Badges for quick visual parsing (EXACT, SEARCH, READ, STATUS, ERROR)
- Tree hierarchy with `├──` / `└──` characters
- File grouping with match counts
- Summary stats in header

---

### Color Output by Default
**Status:** Done (v0.2.7)

Colors are now ON by default for all users. Follows the [no-color.org](https://no-color.org) standard.

- Default: colors enabled
- Opt-out: `NO_COLOR=1` to disable
- Works in Claude Code, terminals, and most environments

**Color scheme:**
| Element | Color |
|---------|-------|
| EXACT badge | Bright magenta |
| SEARCH badge | Bright green |
| READ badge | Bright blue |
| STATUS badge | Bright yellow |
| ERROR badge | Bright red |
| File paths | Cyan |
| Line numbers | Yellow |
| Tree characters | Dim |
| Success text | Green |

---

### Context Command
**Status:** Done (v0.2.8)

Show the entire function/class containing a target line, with smart boundary detection.

```
CONTEXT src/grepl/cli.py:57 ── function index (lines 49-109)

  49 │ def index(path: str, force: bool):
  50 │     """Index a codebase for semantic search."""
  ...
  57 │     # Check prerequisites  ←
  ...
 109 │
```

**Features:**
- Detects function/class boundaries using indentation analysis
- Highlights target line with `←` marker
- Falls back to 50-line context if no block found
- Supports JSON output with `--json`

**Use case:** Claude gets a line number from an error/search but needs the full function context to understand it.

---

## Requested

### `grepl symbols <path>` - Extract code structure
**Status:** Planned
**Complexity:** Medium
**Impact:** High

Extract all function/class definitions from a file or directory. Faster than reading whole files to understand structure.

```
SYMBOLS src/grepl/cli.py ── 12 symbols

├── Functions (8)
│   ├── 39: main()
│   ├── 47: index(path, force)
│   ├── 108: search_cmd(query, limit, json_output, path)
│   └── ...
├── Classes (0)
└── Imports (6)
    ├── click, rich, pathlib
    └── ...
```

**Use case:** Claude needs to understand "what's in this file" before editing.

---

### `grepl refs <symbol>` - Find all usages
**Status:** Planned
**Complexity:** Hard
**Impact:** Very High

Find all references to a symbol (function, class, variable) across the codebase.

```
REFS "format_error" ── 5 references

├── Definition
│   └── src/grepl/utils/tree_formatter.py:198
└── Usages (4)
    ├── src/grepl/cli.py:57: format_error("Ollama is not running"...
    ├── src/grepl/cli.py:61: format_error("Model 'nomic-embed-text'...
    └── ...
```

**Use case:** Before changing a function, Claude needs to know everywhere it's used. Critical for safe refactoring.

---

### `grepl summary` - Codebase overview
**Status:** Planned
**Complexity:** Medium
**Impact:** High

Quick overview of codebase structure, languages, and recent activity.

```
SUMMARY /Users/mike/grepl

├── Structure
│   ├── src/grepl/ (8 files, 1,847 lines)
│   ├── tests/ (3 files, 312 lines)
│   └── docs/ (4 files)
├── Languages
│   ├── Python: 89%
│   └── Markdown: 11%
├── Entry Points
│   └── grepl.cli:main
└── Recent Activity
    ├── cli.py (2 hours ago)
    └── tree_formatter.py (2 hours ago)
```

**Use case:** First thing Claude needs when entering a new codebase. Currently requires multiple commands.

---

### `grepl diff` - Git changes in tree format
**Status:** Planned
**Complexity:** Easy
**Impact:** Medium

Show git changes in tree format with modified functions highlighted.

```
DIFF main..HEAD ── 3 files changed

├── Modified
│   ├── src/grepl/cli.py (+45, -12)
│   │   ├── 47: def index() ← modified
│   │   └── 108: def search_cmd() ← modified
│   └── src/grepl/utils/tree_formatter.py (+368, new file)
└── Deleted
    └── (none)
```

**Use case:** Before committing, Claude needs to understand what changed. Tree format makes it scannable.

---

## Priority Matrix

| Feature | Complexity | Impact | Priority |
|---------|------------|--------|----------|
| Claude Code optimization | Medium | Very High | 1 |
| `symbols` | Medium | High | 2 |
| `summary` | Medium | High | 3 |
| `diff` | Easy | Medium | 4 |
| `refs` | Hard | Very High | 5 |

---

## Requested

### Claude Code Optimization - Enhanced Error Handling & Context
**Status:** Planned
**Complexity:** Medium
**Impact:** Very High

Make grepl the perfect tool for Claude Code by optimizing error messages, adding contextual help, and making every interaction a learning opportunity.

**Error Handling (3-Layer Approach):**
```
ERROR Invalid regex pattern: "User.*{"

What happened:
  Searched for "User.*{" in Python files
  Unmatched opening brace in regex pattern

Why this failed:
  • Regex special characters must be escaped
  • Opening { needs closing } or escape with \

How to fix:
  1. Escape the brace: grepl exact "User.*\{" -p src/
  2. Use exact match: grepl exact "User {" -p src/
  3. Try simpler: grepl exact "User" -p src/

Tip: Run 'grepl --help-regex' for pattern syntax guide
```

**Exit Codes for Claude Understanding:**
- `0` = Success (results found)
- `1` = Pattern syntax error (fix pattern)
- `2` = No matches (valid pattern, nothing found)
- `3` = Path doesn't exist (fix path)
- `4` = Permission denied (check permissions)

**Contextual Tips & Suggestions:**
```
Found 5 results for "UserModel" (23ms, 147 files scanned)

Tip: In Swift projects, also try:
  • grepl semantic "user data model" (concept search)
  • grepl exact "struct User" (structural search)
  • grepl context src/User.swift:42 (view function)

Performance: ✓ Fast (< 50ms)
```

**"Did You Mean?" for Failed Searches:**
```
No results for "asyncFunc.*throws"

Searched: 23 Swift files in /src

Similar patterns that worked:
  • grepl exact "async" -p src/
  • grepl semantic "async functions"
  • grepl exact "throws" -p src/

Try next:
  grepl read src/main.swift  # View file first
  grepl exact "async" -p src/  # Simpler pattern
```

**Advanced Usage Hints:**
- Add `--tips` flag to show contextual usage tips
- Add `--advanced` flag to show advanced patterns
- Include performance metrics (files scanned, time taken)
- Show search scope in every output
- Suggest related patterns based on codebase

**Implementation Checklist:**
- [ ] 3-layer error messages (problem, context, fix)
- [ ] Meaningful exit codes (0-4)
- [ ] "Did you mean?" suggestions
- [ ] Performance metrics in output
- [ ] `--tips` and `--advanced` flags
- [ ] Contextual suggestions based on language
- [ ] Show search scope (files/paths)
- [ ] Results truncation warnings
- [ ] Command suggestions on errors
- [ ] Pattern syntax guide (`--help-regex`)

**Use case:** Claude needs to never get stuck. Every error should teach Claude what to do next. Every success should show related patterns. This makes grepl self-documenting and maximizes Claude Code effectiveness.

---

## Ideas / Backlog

Space for future feature ideas:

- **Fuzzy search** - For when you don't remember the exact name
- **Type filters** - Search only in specific file types (`--type py`)
- **TODO finder** - Find all TODOs, FIXMEs, HACKs in codebase
- **Dependency graph** - Show what imports what
- **Change impact** - "If I change this function, what might break?"
- **Token-efficient mode** - Output format optimized for LLM context windows
