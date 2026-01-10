# ast-grep Integration Plan for Grepl

This document outlines how to add **AST/structure-aware search** to grepl using **ast-grep**. The goal is to complement:

- `grepl exact` (text/regex via ripgrep)
- `grepl search` (semantic via ChromaDB + embeddings)
- `grepl find` (hybrid merge/rerank)

with a new **structural engine** powered by tree-sitter patterns.

---

## Goals

- Add a **structural search** command that can precisely match code shapes (AST patterns).
- Support **semantic → AST** narrowing by default (fast, ergonomic).
- Support **exhaustive AST scans** for refactors/policy enforcement.
- Parse `ast-grep --json` output and format results consistently with the rest of grepl.

## Non-goals (initially)

- Bundling ast-grep into grepl as a Python dependency.
- Writing/maintaining large rule packs (YAML) inside this repo.

---

## Dependency Strategy

Treat ast-grep as an **optional external binary**.

- Detect on PATH via `shutil.which("ast-grep")`.
- Provide install hints in the error message:
  - `brew install ast-grep`
  - `npm i -g @ast-grep/cli`
  - (optional) `cargo install ast-grep`

This keeps grepl’s Python dependency surface unchanged and avoids platform-specific tree-sitter binding issues.

---

## CLI UX

### 1) New command: `grepl ast`

Minimal, one-shot structural search using `ast-grep run`:

```bash
grepl ast -l ts -p src --pattern 'console.log($A)'
grepl ast -l py -p . --pattern 'except $E: $$$'
```

Recommended flags:

- `--pattern, -P` (required): AST pattern
- `--lang, -l` (required): language (`ts`, `js`, `py`, `rs`, …)
- `--path, -p` (default `.`): directory to scan
- `--globs` (optional): forward to ast-grep for file filtering
- `--json`: machine output

### 2) Semantic narrowing (default): `--semantic` (+ `--top` budget)

Use grepl’s semantic index to pick candidate files, then run ast-grep only on those files.

```bash
grepl ast -l ts --pattern 'fetch($URL, $$$)' \
  --semantic 'api client' --top 75
```

Suggested options:

- `--semantic <query>`: run `grepl search` internally (library call) to select candidates
- `--top <n>` (default ~50–100): how many top semantic files to include
- `--score-threshold <f>` (optional): include any file above a semantic score cutoff

Output should disclose coverage:

> AST applied to 47 files (semantic narrowed from 12,391). Use `--exhaustive` to scan all.

### 3) Exhaustive mode: `--exhaustive`

Scan the full `--path` with ast-grep (completeness-first):

```bash
grepl ast -l ts -p src --pattern 'useEffect($$$)' --exhaustive
```

### 4) Rewrites (codemod) – **dry-run by default**

ast-grep supports rewrites directly:

```bash
# show changes (no edits)
grepl ast -l ts -p src \
  --pattern 'foo($A)' --rewrite 'bar($A)' --dry-run

# apply edits in-place
grepl ast -l ts -p src \
  --pattern 'foo($A)' --rewrite 'bar($A)' --apply
```

Recommended safety:

- Default `--dry-run` if `--rewrite` is provided.
- Only modify files when `--apply` is explicitly set.

---

## Execution Model

### Command invocation

Use `ast-grep run` (one-shot inline pattern) with JSON output:

```bash
ast-grep run \
  --pattern '<PATTERN>' \
  --lang <LANG> \
  --json \
  [--rewrite '<REWRITE>'] \
  [--globs '<GLOBS>'] \
  [PATHS...]
```

Implementation notes:

- Prefer passing explicit candidate file paths as `PATHS...` when doing semantic narrowing.
- Batch candidates (e.g. 50–200 per process) to avoid enormous argv size.
- Always capture stdout/stderr; non-zero exit should map to grepl’s `ExitCode` + rich error formatting.

### Candidate selection (semantic → AST)

When `--semantic` is provided:

1. Call grepl’s semantic `search(project_path, query, limit=...)`.
2. Convert hits → unique file paths.
3. Run ast-grep on those file paths.

Optional “safety net” candidates (cheap + useful):

- Include files from `git diff --name-only` (if available) or “recently modified” within N days.
- Include files from a quick identifier grep when the query looks like a symbol.
- Include “hot” directories (configurable) like `auth/`, `core/`, `infra/`.

---

## Result Normalization

ast-grep `--json` emits an array of match objects (see official JSON mode docs). For grepl, normalize into a common schema similar to `exact/search/find`:

```json
{
  "source": "ast",
  "file": "src/foo.ts",
  "range": {"start": {"line": 10, "column": 2}, "end": {"line": 10, "column": 18}},
  "text": "console.log(x)",
  "language": "TypeScript",
  "metaVariables": {"single": {"A": {"text": "x"}}},
  "replacement": "console.info(x)"
}
```

Notes:

- ast-grep uses **0-based** line/column in JSON; grepl typically displays **1-based** line numbers. Convert for display.
- Keep the raw ast-grep payload (or selected fields) when `--json` is requested.
- For rich output, print a compact snippet + `file:line` like grepl’s other commands.

---

## Where to Put the Code

Suggested minimal additions:

- `src/grepl/ast_engine.py`
  - `check_ast_grep_installed()`
  - `run_ast_grep(pattern, lang, paths, rewrite=None, globs=None) -> list[AstMatch]`
- `src/grepl/cli.py`
  - add `@main.command()` for `ast`
  - wire semantic candidate selection via existing store/search functions
- `src/grepl/utils/tree_formatter.py`
  - add `format_ast_results(...)` for rich output, plus JSON pass-through via existing `format_json_output`

Keep all subprocess handling in one place (engine module) so tests can mock it cleanly.

---

## Error Handling

Map common failures to actionable messages:

- ast-grep not installed → show install commands + `which` hint
- invalid pattern → surface stderr + suggest `ast-grep run --debug-query`
- missing/unknown language → list common `--lang` values and point to ast-grep docs
- huge candidate list → warn and suggest reducing `--top` or using `--globs`

---

## Testing Strategy

Add tests that don’t require ast-grep installed:

- Unit tests for argument construction + JSON parsing:
  - mock `subprocess.run` to return fixed JSON output
  - verify normalization (0-based → 1-based for display)
  - verify exit-code handling and rich error formatting

- Lightweight integration test:
  - create a temp repo with a few files
  - provide a fake `ast-grep` shim executable earlier on PATH that prints deterministic JSON
  - run the click command and assert formatted output

---

## Recommended Defaults

- Default mode when `--semantic` is present: **semantic → AST**
- Default `--top`: **50–100** files
- Require explicit `--exhaustive` for full scans
- Require explicit `--apply` for rewrites
