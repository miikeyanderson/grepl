---
name: code-search
description: Semantic code search for finding code by meaning. Use when searching for concepts, logic, patterns, or asking "where is X handled" or "find code that does Y".
allowed-tools: Bash(grepl:*)
---

# Code Search Skill

## When to Use This Skill

Use `grepl search` for:
- Finding code by concept ("authentication logic", "error handling")
- Exploring unfamiliar codebases
- Searching by intent, not exact text

Use `grepl exact` for:
- Specific strings, function names, imports
- TODOs, FIXMEs, exact patterns

Use `grepl read` for:
- Reading file contents after finding a match
- Viewing context around a specific line

## Commands

### Index (first time only)
```bash
grepl index .
```

### Semantic Search
```bash
grepl search "your query" -n 10
```

### Exact Match
```bash
grepl exact "pattern"
```

### Read File
```bash
grepl read file.py:45       # ~50 lines centered on line 45
grepl read file.py:30-80    # Lines 30-80
grepl read file.py -c 100   # 100 lines of context
```

### Check Status
```bash
grepl status
```

## Examples

| Task | Command |
|------|---------|
| Find auth logic | `grepl search "authentication"` |
| Find error handling | `grepl search "error handling patterns"` |
| Find specific function | `grepl exact "def processPayment"` |
| Find all TODOs | `grepl exact "TODO"` |
| Read around line 45 | `grepl read src/auth.py:45` |

## Workflow

1. Check if index exists: `grepl status`
2. If not indexed: `grepl index .`
3. Search: `grepl search "your query"`
4. Read context: `grepl read file.py:45`

## Output Format

Results show file:line with matched content:
```
src/auth/login.ts:45: async function validateUser(token) {
src/auth/login.ts:46:   const decoded = jwt.verify(token);
--
src/middleware/auth.ts:12: export const requireAuth = ...
```
