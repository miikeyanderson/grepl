"""Grepl CLI - Semantic code search."""

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .embedder import (
    check_ollama,
    check_model,
    get_backend_info,
    list_available_backends,
    set_preferred_backend,
)
from .chunker import chunk_codebase, chunk_file, collect_file_metadata
from .store import (
    index_chunks,
    search,
    clear_index,
    get_stats,
    get_detailed_stats,
    has_index,
    check_semantic_ready,
    store_index_metadata,
    get_changed_files,
    get_collection,
)
from .planner import analyze_query, Strategy, ExecutionPlan
from .ranker import Hit, RankWeights, merge_results, rerank
from .ast_grep import (
    check_ast_grep,
    format_ast_grep_install_hint,
    run_ast_grep_multi,
    AstGrepConfig,
    resolve_rule_file,
    list_available_rules,
)
from .utils.formatters import format_json_output
from .utils.tree_formatter import (
    format_exact_results,
    format_search_results,
    format_find_results,
    format_read_output,
    format_context_output,
    format_status_output,
    format_ls_output,
    format_index_header,
    format_index_progress,
    format_error,
    format_error_rich,
    format_no_results,
    format_success_header,
    format_tip,
    format_performance,
    ExitCode,
    badge,
    cyan,
    green,
    yellow,
    dim,
    Colors,
)

console = Console()


class GreplGroup(click.Group):
    """Custom Click group with formatted help output."""

    def format_help(self, ctx, formatter):
        """Override to print custom formatted help."""
        self.format_custom_help()

    def format_custom_help(self):
        """Print beautifully formatted help."""
        # Header
        print(f"\n{badge('GREPL', Colors.BRIGHT_CYAN)} {dim('v' + __version__)} {dim('─')} {dim('Explainable code search for humans and AI')}\n")

        # Commands section
        print(f"  {Colors.BOLD}Commands{Colors.RESET}\n")

        commands = [
            ("find", "Hybrid + AST search", "grepl find \"error\" --ast \"try { $$ }\"", True),
            ("exact", "Pattern search (ripgrep)", "grepl exact \"def main\" -p src/", False),
            ("search", "Semantic search", "grepl search \"auth logic\"", False),
            ("read", "Read file with context", "grepl read src/cli.py:50", False),
            ("context", "Show function at line", "grepl context src/cli.py:100", False),
            ("ls", "List directory with tree structure", "grepl ls src/", False),
            ("index", "Index codebase", "grepl index", False),
            ("status", "Check index status", "grepl status", False),
            ("model", "Manage embedding models", "grepl model", False),
            ("clear", "Clear search index", "grepl clear", False),
        ]

        for cmd, desc, example, is_primary in commands:
            prefix = "├──" if cmd != "clear" else "└──"
            if is_primary:
                print(f"  {dim(prefix)} {green(cmd):12} {desc}")
                print(f"  {dim('│')}   {dim('Example:')} {cyan(example)}")
            else:
                print(f"  {dim(prefix)} {cyan(cmd):12} {dim(desc)}")

        # Quick examples section
        print(f"\n  {Colors.BOLD}Quick Start{Colors.RESET}\n")
        q = '"'
        print(f"  {dim('$')} {cyan('grepl index')}                    {dim('# Index current directory')}")
        print(f"  {dim('$')} {cyan(f'grepl find {q}authentication{q}')}   {dim('# Hybrid search')}")
        print(f"  {dim('$')} {cyan(f'grepl exact {q}TODO{q} -i')}         {dim('# Case-insensitive grep')}")
        print(f"  {dim('$')} {cyan('grepl read file.py:100')}         {dim('# Read around line 100')}")

        # AST examples section
        print(f"\n  {Colors.BOLD}AST Search{Colors.RESET} {dim('(requires: brew install ast-grep)')}\n")
        ast_ex1 = 'grepl find --ast "print($$$)" --ast-lang swift'
        ast_ex2 = 'grepl find --ast-rule swift-no-print --exhaustive'
        ast_ex3 = 'grepl find "error" --ast "catch { $$ }" --plan'
        print(f"  {dim('$')} {cyan(ast_ex1)}")
        print(f"  {dim('$')} {cyan(ast_ex2)}")
        print(f"  {dim('$')} {cyan(ast_ex3)}")

        # Footer
        print(f"\n  {dim('Run')} {cyan('grepl <command> --help')} {dim('for detailed options')}\n")


@click.group(cls=GreplGroup)
@click.version_option(version=__version__)
def main():
    """Grepl - Semantic code search powered by ChromaDB + Ollama."""
    pass


@main.command()
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("--force", "-f", is_flag=True, help="Force full reindex")
def index(path: str, force: bool):
    """Index a codebase for semantic search.

    Uses incremental indexing by default - only indexes new/modified files.
    Use --force for a full reindex.
    """
    project_path = Path(path).resolve()

    # Print header
    print(format_index_header(project_path))
    print()

    # Check prerequisites
    if not check_ollama():
        format_error("Ollama is not running", hint=f"Start with: {cyan('ollama serve')}")
        sys.exit(1)

    if not check_model():
        format_error("Model 'nomic-embed-text' not found", hint=f"Pull with: {cyan('ollama pull nomic-embed-text')}")
        sys.exit(1)

    # Collect metadata for all current files
    format_index_progress("Scanning files...")
    current_files_metadata = collect_file_metadata(project_path)

    if not current_files_metadata:
        print(f"  {yellow('!')} No files found to index")
        return

    # Determine what needs to be indexed
    if force:
        # Full reindex
        format_index_progress("Full reindex requested")
        clear_index(project_path)
        files_to_index = list(current_files_metadata.keys())
        new_count, modified_count, deleted_count = len(files_to_index), 0, 0
    elif has_index(project_path):
        # Incremental update
        new_files, modified_files, deleted_files = get_changed_files(
            project_path, current_files_metadata
        )
        files_to_index = new_files + modified_files
        new_count, modified_count, deleted_count = len(new_files), len(modified_files), len(deleted_files)

        if not files_to_index and not deleted_files:
            stats = get_stats(project_path)
            print(f"  {green('✓')} Index is up to date ({cyan(str(stats['chunks']))} chunks)")
            return

        # Show what will be updated
        if new_count > 0:
            format_index_progress(f"New files: {new_count}")
        if modified_count > 0:
            format_index_progress(f"Modified files: {modified_count}")
        if deleted_count > 0:
            format_index_progress(f"Deleted files: {deleted_count}")

        # Delete chunks for removed files
        if deleted_files:
            collection = get_collection(project_path)
            for deleted_file in deleted_files:
                try:
                    # Get all chunks for this file
                    results = collection.get(where={"file_path": deleted_file})
                    if results and results.get("ids"):
                        collection.delete(ids=results["ids"])
                except Exception:
                    pass
            format_index_progress(f"Removed {deleted_count} deleted files", done=True)

        # Delete chunks for modified files (they will be re-indexed)
        if modified_files:
            collection = get_collection(project_path)
            for modified_file in modified_files:
                try:
                    results = collection.get(where={"file_path": modified_file})
                    if results and results.get("ids"):
                        collection.delete(ids=results["ids"])
                except Exception:
                    pass
    else:
        # First time indexing
        files_to_index = list(current_files_metadata.keys())
        new_count, modified_count, deleted_count = len(files_to_index), 0, 0

    # Chunk files that need indexing
    format_index_progress(f"Chunking {len(files_to_index)} files...")
    chunks = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Chunking...", total=None)
        for file_path_str in files_to_index:
            file_chunks = chunk_file(Path(file_path_str))
            chunks.extend(file_chunks)
        progress.update(task, description=f"Found {len(chunks)} chunks")

    format_index_progress(f"Found {len(chunks)} chunks", done=True)

    if not chunks and deleted_count == 0:
        print(f"  {yellow('!')} No chunks to index")
        return

    # Index chunks
    if chunks:
        format_index_progress("Generating embeddings...")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Indexing...", total=None)
            # Use incremental indexing - add new chunks without clearing
            from .store import _index_batch
            collection = get_collection(project_path)
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                _index_batch(collection, batch, project_path)
            progress.update(task, description="Done")

        format_index_progress(f"Indexed {len(chunks)} chunks", done=True)

    # Store metadata for next incremental update
    store_index_metadata(project_path, current_files_metadata)

    # Show summary
    print()
    if force or not has_index(project_path):
        print(f"  {green('Ready!')} Run {cyan('grepl search <query>')} to search")
    else:
        summary_parts = []
        if new_count > 0:
            summary_parts.append(f"{new_count} new")
        if modified_count > 0:
            summary_parts.append(f"{modified_count} modified")
        if deleted_count > 0:
            summary_parts.append(f"{deleted_count} deleted")
        print(f"  {green('Updated!')} {', '.join(summary_parts)}")


@main.command("search")
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Number of results (default: 10)")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
@click.option("--path", "-p", default=".", help="Path to search")
def search_cmd(query: str, limit: int, json_output: bool, path: str):
    """Semantic search across indexed codebase."""
    project_path = Path(path).resolve()

    # Check prerequisites
    if not check_ollama():
        if json_output:
            print(json.dumps({"error": "Ollama is not running"}))
        else:
            format_error("Ollama is not running", hint=f"Start with: {cyan('ollama serve')}")
        sys.exit(1)

    if not has_index(project_path):
        if json_output:
            print(json.dumps({"error": "Codebase not indexed"}))
        else:
            format_error("Codebase not indexed", hint=f"Run: {cyan(f'grepl index {path}')}")
        sys.exit(1)

    results = search(project_path, query, limit)

    if json_output:
        format_json_output(results, raw=True)
        return

    format_search_results(results, query, max_lines=3)


def _lang_to_exts(lang: Optional[str]) -> Optional[Set[str]]:
    if not lang:
        return None
    mapping = {
        "py": {".py"},
        "python": {".py"},
        "js": {".js", ".jsx"},
        "javascript": {".js", ".jsx"},
        "ts": {".ts", ".tsx"},
        "typescript": {".ts", ".tsx"},
        "swift": {".swift"},
        "go": {".go"},
        "rs": {".rs"},
        "rust": {".rs"},
        "rb": {".rb"},
        "ruby": {".rb"},
        "java": {".java"},
        "kt": {".kt"},
        "kotlin": {".kt"},
        "c": {".c", ".h"},
        "cpp": {".cpp", ".hpp", ".cc", ".hh", ".h"},
        "md": {".md"},
    }

    exts: Set[str] = set()
    for part in re.split(r"[ ,]+", lang.strip().lower()):
        if not part:
            continue
        if part in mapping:
            exts |= mapping[part]
        elif part.startswith(".") and len(part) <= 8:
            exts.add(part)
    return exts or None


def _extract_symbols(text: str) -> List[str]:
    symbols: List[str] = []
    for m in re.finditer(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE):
        name = m.group(1)
        if name not in symbols:
            symbols.append(name)
        if len(symbols) >= 5:
            break
    return symbols


def _grep_matches_to_hits(
    matches: List[dict],
    *,
    grep_score: float = 1.0,
) -> List[Hit]:
    hits: List[Hit] = []
    for m in matches:
        file_path = m["file_path"]
        line = m["line"]
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception:
            continue

        total = len(lines)
        bounds = find_block_bounds(lines, line)
        if bounds is None:
            half = 15
            start_line = max(1, line - half)
            end_line = min(total, line + half)
        else:
            start_line, end_line, _, _ = bounds

        preview = "".join(lines[start_line - 1:end_line]).rstrip("\n")
        symbols = _extract_symbols(preview)
        hits.append(
            Hit(
                source="grep",
                file_path=file_path,
                start_line=int(start_line),
                end_line=int(end_line),
                score=0.0,
                preview=preview,
                symbols=symbols,
                grep_score=float(grep_score),
                semantic_score=0.0,
            )
        )
    return hits


def _run_rg(pattern: str, search_path: Path, *, fixed: bool, max_results: int, exts: Optional[Set[str]]) -> List[dict]:
    cmd = ["rg", "-n", "--color=never", "--no-heading"]
    if fixed:
        cmd.append("-F")
    if exts:
        for ext in sorted(exts):
            cmd.extend(["--glob", f"*{ext}"])
    cmd.extend([pattern, str(search_path)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        return []
    out = result.stdout
    matches: List[dict] = []
    for raw in out.splitlines():
        if ":" not in raw:
            continue
        fp, line_str, rest = raw.split(":", 2)
        if not line_str.isdigit():
            continue
        matches.append({"file_path": fp, "line": int(line_str), "content": rest})
        if len(matches) >= max_results:
            break
    return matches


@main.command("find")
@click.argument("query", default="")
@click.option("-k", "-n", "--top-k", default=10, show_default=True, help="Number of results")
@click.option("-p", "--path", default=".", help="Search path")
@click.option("--lang", default=None, help="Limit to languages (e.g. py,ts,swift)")
@click.option("--grep-only", is_flag=True, help="Only use grep")
@click.option("--semantic-only", is_flag=True, help="Only use semantic")
@click.option("--precise", is_flag=True, help="Require grep confirmation")
@click.option("--ast", "ast_patterns", multiple=True, help="AST pattern (repeatable)")
@click.option("--ast-rule", "ast_rules", multiple=True, help="AST rule file (repeatable)")
@click.option("--ast-lang", default=None, help="Language for AST parsing")
@click.option("--exhaustive", is_flag=True, help="Run AST on entire repo (slow)")
@click.option("--ast-top-files", default=100, show_default=True, help="Max files to scan with AST")
@click.option("--ast-max-matches", default=500, show_default=True, help="Max AST matches to return")
@click.option("--ast-optional", is_flag=True, help="Skip AST if sg not installed (warn instead of error)")
@click.option("--strategy", type=click.Choice(["explore", "codemod", "grep"]), default=None, help="Search strategy preset")
@click.option("--plan", "show_plan", is_flag=True, help="Show execution plan without running")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
def find_cmd(
    query: str,
    top_k: int,
    path: str,
    lang: Optional[str],
    grep_only: bool,
    semantic_only: bool,
    precise: bool,
    ast_patterns: tuple,
    ast_rules: tuple,
    ast_lang: Optional[str],
    exhaustive: bool,
    ast_top_files: int,
    ast_max_matches: int,
    ast_optional: bool,
    strategy: Optional[str],
    show_plan: bool,
    json_output: bool,
):
    """Hybrid search: combines exact matching (ripgrep), semantic search (ChromaDB), and AST (ast-grep).

    Modes:
      • Auto (default): chooses grep vs semantic vs hybrid based on the query.
      • --grep-only: exact matches only (no indexing required).
      • --semantic-only: semantic matches only (requires: grepl index + Ollama).
      • --ast: structural code search (requires: ast-grep installed).

    Strategies:
      • explore: semantic → grep → ast (default when --ast provided)
      • codemod: ast exhaustive (for refactoring)
      • grep: grep only (fast)

    Examples:
      grepl find "error handling" -k 5
      grepl find "TopNavBar" --grep-only
      grepl find "authentication flow" --semantic-only

      # AST search (structural)
      grepl find "retry logic" --ast "try { $$ } catch { $$ }" --ast-lang swift
      grepl find --ast "DispatchQueue.main.async { $$ }" --ast-lang swift

      # Strategy presets
      grepl find --strategy codemod --ast "print($$)" --ast-lang swift  # Full repo AST
      grepl find "error" --strategy explore --ast "catch { $$ }"        # Narrow then AST

      # Show plan without executing
      grepl find "auth" --ast "guard let $$ else { return }" --plan
    """
    start_time = time.time()
    project_path = Path(path).resolve()

    # Validate path exists
    if not project_path.exists():
        if json_output:
            print(json.dumps({
                "error": "path_not_found",
                "message": f"Path does not exist: {path}",
                "path": str(project_path),
            }))
        else:
            format_error_rich(
                f"Path not found: {path}",
                context=f"Tried to search in '{project_path}'",
                why=["The specified path does not exist", "Path may be misspelled or relative to wrong directory"],
                fixes=[
                    f"grepl find {repr(query)} -p .",
                    f"ls -la {project_path.parent}" if project_path.parent != project_path else "ls -la .",
                ],
                tip="Use '.' to search current directory, or provide full path",
            )
        sys.exit(ExitCode.PATH_ERROR)

    exts = _lang_to_exts(lang)

    # Resolve rule files before building plan
    resolved_rules: List[str] = []
    rule_sources: Dict[str, str] = {}
    for rule_name in ast_rules:
        resolved_path, source = resolve_rule_file(rule_name, project_path)
        if resolved_path:
            resolved_rules.append(str(resolved_path))
            rule_sources[rule_name] = source
        else:
            if not json_output:
                format_error(f"Rule not found: {rule_name}")
                available = list_available_rules(project_path)
                all_rules = available["project"] + available["user"] + available["builtin"]
                if all_rules:
                    print(f"  {dim('Available rules:')} {', '.join(all_rules)}")
            else:
                print(json.dumps({"error": f"Rule not found: {rule_name}"}))
            sys.exit(1)

    # Build query plan with AST options
    plan = analyze_query(
        query,
        grep_only=grep_only,
        semantic_only=semantic_only,
        precise=precise,
        ast_patterns=list(ast_patterns),
        ast_rules=tuple(resolved_rules),
        ast_language=ast_lang,
        ast_exhaustive=exhaustive,
        strategy=strategy,  # type: ignore
    )
    mode = plan.mode

    # Create ExecutionPlan for richer output
    exec_plan = ExecutionPlan(
        query_plan=plan,
        query=query,
        path=path,
        ast_top_files=ast_top_files,
        ast_max_matches=ast_max_matches,
    )

    # Add reasoning for each stage
    if plan.run_semantic:
        exec_plan.add_reason("semantic", "Query contains natural language or concept terms")
    if plan.run_grep:
        if plan.grep_fixed:
            exec_plan.add_reason("grep", f"Exact match for quoted term: {plan.grep_pattern}")
        else:
            exec_plan.add_reason("grep", f"Pattern search for identifier: {plan.grep_pattern}")
    if plan.run_ast:
        if plan.ast_exhaustive:
            exec_plan.add_reason("ast", "Exhaustive mode: scanning entire repository")
        else:
            exec_plan.add_reason("ast", "Narrow mode: will scan files from grep/semantic hits")

    # Handle --plan flag (dry run)
    if show_plan:
        if json_output:
            format_json_output(exec_plan.to_dict(), raw=True)
        else:
            print(f"\n{badge('PLAN', Colors.BRIGHT_YELLOW)} {dim('Execution plan for query')}\n")
            for line in exec_plan.format_human().split("\n"):
                print(f"  {line}")
            print()
        return

    # Check AST dependencies if needed
    skip_ast = False
    if plan.run_ast and not check_ast_grep():
        if ast_optional:
            # Warn but continue without AST
            skip_ast = True
            if not json_output:
                print(f"  {yellow('!')} ast-grep (sg) not installed - skipping AST stage")
                print(f"    {dim('Install with:')} {cyan('brew install ast-grep')}")
                print()
        else:
            if json_output:
                print(json.dumps({"error": "ast-grep (sg) not installed", "hint": "brew install ast-grep"}))
            else:
                format_error("ast-grep (sg) not installed")
                print(format_ast_grep_install_hint())
            sys.exit(1)

    # For natural-language queries, treat grep matches as a weaker signal than semantic.
    # This prevents a single extracted token from dominating the ranking.
    word_count = len(re.findall(r"[A-Za-z0-9_]+", query))
    grep_hit_score = 1.0
    if (
        not grep_only
        and not semantic_only
        and plan.run_semantic
        and word_count >= 4
        and not plan.grep_fixed
    ):
        grep_hit_score = 0.55

    grep_hits: List[Hit] = []
    semantic_hits: List[Hit] = []
    ast_hits: List[Hit] = []

    # Track semantic search availability for degradation messaging
    semantic_skipped = False
    semantic_skip_reason = None

    # Grep
    if plan.run_grep and plan.grep_pattern:
        max_grep = max(50, min(500, top_k * 25))
        matches = _run_rg(plan.grep_pattern, project_path, fixed=plan.grep_fixed, max_results=max_grep, exts=exts)
        grep_hits = _grep_matches_to_hits(matches, grep_score=grep_hit_score)

    # Semantic
    if plan.run_semantic and plan.semantic_query:
        if not check_ollama():
            semantic_skipped = True
            semantic_skip_reason = "ollama_not_running"
            if semantic_only and not grep_hits:
                if json_output:
                    print(json.dumps({"error": "Ollama is not running"}))
                else:
                    format_error("Ollama is not running", hint=f"Start with: {cyan('ollama serve')}")
                sys.exit(1)
        elif not has_index(project_path):
            semantic_skipped = True
            semantic_skip_reason = "not_indexed"
            if semantic_only and not grep_hits:
                if json_output:
                    print(json.dumps({"error": "Codebase not indexed"}))
                else:
                    format_error("Codebase not indexed", hint=f"Run: {cyan(f'grepl index {path}')}")
                sys.exit(1)
        else:
            raw = search(project_path, plan.semantic_query, max(10, top_k * 3))
            if exts:
                raw = [r for r in raw if Path(r.get("file_path", "")).suffix in exts]
            for r in raw:
                content = r.get("content", "")
                meta_symbols = r.get("symbols") or []
                semantic_hits.append(
                    Hit(
                        source="semantic",
                        file_path=r.get("file_path", "unknown"),
                        start_line=int(r.get("start_line", 1)),
                        end_line=int(r.get("end_line", r.get("start_line", 1))),
                        score=0.0,
                        preview=content,
                        symbols=list(meta_symbols) if meta_symbols else _extract_symbols(content),
                        grep_score=0.0,
                        semantic_score=float(r.get("score", 0.0)),
                    )
                )

    # AST (ast-grep)
    if plan.run_ast and not skip_ast:
        ast_config = AstGrepConfig(
            patterns=list(plan.ast_patterns),
            rule_files=list(plan.ast_rules),
            language=plan.ast_language,
            exhaustive=plan.ast_exhaustive,
        )

        # Determine file set for AST: prefer grep hits > semantic candidates > exhaustive
        ast_files: Optional[List[str]] = None
        if not plan.ast_exhaustive:
            if grep_hits:
                # Narrow to files that had grep matches, capped by ast_top_files
                all_files = list(set(h.file_path for h in grep_hits))
                ast_files = all_files[:ast_top_files]
            elif semantic_hits:
                # Narrow to files from semantic search, capped by ast_top_files
                all_files = list(set(h.file_path for h in semantic_hits))
                ast_files = all_files[:ast_top_files]

        ast_hits = run_ast_grep_multi(
            config=ast_config,
            files=ast_files,
            root_path=project_path,
        )

        # Cap the number of AST matches returned
        if len(ast_hits) > ast_max_matches:
            ast_hits = ast_hits[:ast_max_matches]

    # Merge + rank
    merged = merge_results(grep_hits, semantic_hits, ast_hits, overlap_lines=3)
    ranked = rerank(merged, weights=RankWeights(), max_per_file=3, ast_exhaustive=plan.ast_exhaustive)

    # Reflect what actually ran/returned.
    effective_mode = mode
    sources_present = []
    if grep_hits:
        sources_present.append("grep")
    if semantic_hits:
        sources_present.append("semantic")
    if ast_hits:
        sources_present.append("ast")

    if len(sources_present) == 1:
        effective_mode = sources_present[0] if sources_present[0] != "grep" else "exact"
    elif len(sources_present) >= 2:
        effective_mode = "hybrid"

    if precise:
        ranked_precise = [h for h in ranked if h.grep_score > 0 or h.ast_score > 0]
        if ranked_precise:
            ranked = ranked_precise

    ranked = ranked[: max(0, top_k)]

    elapsed_ms = int((time.time() - start_time) * 1000)
    if json_output:
        payload = {
            "query": query,
            "mode": effective_mode,
            "plan": plan.describe(),
            "time_ms": elapsed_ms,
            "semantic_available": not semantic_skipped,
            "semantic_skip_reason": semantic_skip_reason,
            "results": [
                {
                    "source": h.source,
                    "file_path": h.file_path,
                    "start_line": h.start_line,
                    "end_line": h.end_line,
                    "score": h.score,
                    "grep_score": h.grep_score,
                    "semantic_score": h.semantic_score,
                    "ast_score": h.ast_score,
                    "ast_pattern": h.ast_pattern,
                    "preview": h.preview,
                    "symbols": h.symbols,
                }
                for h in ranked
            ],
        }
        format_json_output(payload, raw=True)
        return

    if not ranked:
        # Provide diagnostic context about WHY no matches
        diagnostics = []
        suggestions = []

        # Check what searches actually ran
        if plan.run_grep:
            if not grep_hits:
                diagnostics.append("Grep: No pattern matches found")
            else:
                diagnostics.append(f"Grep: {len(grep_hits)} initial matches (filtered out)")

        if plan.run_semantic:
            if not check_ollama():
                diagnostics.append("Semantic: Ollama not running")
                suggestions.append("ollama serve")
            elif not has_index(project_path):
                diagnostics.append(f"Semantic: Path not indexed")
                suggestions.append(f"grepl index {path}")
            elif not semantic_hits:
                diagnostics.append("Semantic: No relevant code found")

        if plan.run_ast:
            if not ast_hits:
                diagnostics.append("AST: No structural matches found")

        # Build suggestions based on what failed
        if plan.grep_pattern:
            suggestions.append(f"grepl exact {repr(plan.grep_pattern)} -p {path}")
        if plan.semantic_query:
            suggestions.append(f"grepl search {repr(query)} -p {path}")

        # Format output with diagnostics
        if not json_output:
            format_no_results(
                pattern=query,
                search_type="FIND",
                time_ms=elapsed_ms,
                suggestions=suggestions
            )
            if diagnostics:
                console.print(f"\n  [dim]Diagnostics:[/dim]")
                for diag in diagnostics:
                    console.print(f"    [dim]• {diag}[/dim]")
        else:
            print(json.dumps({
                "query": query,
                "matches": 0,
                "time_ms": elapsed_ms,
                "diagnostics": diagnostics,
                "suggestions": suggestions,
            }))

        sys.exit(ExitCode.NO_MATCHES)

    format_find_results(
        [
            {
                "source": h.source,
                "file_path": h.file_path,
                "start_line": h.start_line,
                "end_line": h.end_line,
                "score": h.score,
                "preview": h.preview,
                "symbols": h.symbols,
            }
            for h in ranked
        ],
        query=query,
        mode=effective_mode,
        max_lines=3,
    )

    # Show degradation message if semantic was skipped but we found results
    if semantic_skipped and ranked:
        from .utils.tree_formatter import yellow, cyan, dim
        reason_msg = {
            "ollama_not_running": f"Ollama not running. Start with {cyan('ollama serve')} for semantic search.",
            "not_indexed": f"Codebase not indexed. Run {cyan(f'grepl index {path}')} for semantic search.",
        }.get(semantic_skip_reason, f"Semantic search unavailable ({semantic_skip_reason})")

        console.print(f"\n  {yellow('ℹ')} Note: {dim(reason_msg)}")


@main.command()
@click.argument("pattern")
@click.option("--limit", "-n", default=None, type=int, help="Max number of results")
@click.option("--ignore-case", "-i", is_flag=True, help="Case-insensitive search")
@click.option("--path", "-p", default=".", help="Path to search")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
def exact(pattern: str, limit: int, ignore_case: bool, path: str, json_output: bool):
    """Exact pattern search (uses ripgrep/grep)."""
    start_time = time.time()
    search_path = Path(path).resolve()

    # Check if path exists
    if not search_path.exists():
        if json_output:
            print(json.dumps({"error": f"Path not found: {path}"}))
        else:
            format_error_rich(
                f"Path not found: {path}",
                context=f"Tried to search in '{path}'",
                why=["The specified path does not exist", "Path may be misspelled"],
                fixes=[
                    f"grepl exact {repr(pattern)} -p .",
                    f"ls {path.rsplit('/', 1)[0] if '/' in path else '.'}",
                ],
                tip="Use '.' to search current directory",
            )
        sys.exit(ExitCode.PATH_ERROR)

    try:
        # Try ripgrep first
        # -H: always include filename (even for single files)
        # -n: show line numbers
        # --no-heading: output file:line:content format (no grouped headers)
        cmd = ["rg", "-H", "-n", "--color=never", "--no-heading", pattern, str(search_path)]
        if ignore_case:
            cmd.insert(4, "-i")
        if limit:
            cmd.insert(4, f"--max-count={limit}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            output = result.stdout
            if limit:
                lines = output.strip().split("\n")
                output = "\n".join(lines[:limit])

            # Parse results
            matches_by_file: Dict[str, List[dict]] = {}
            for line in output.strip().split("\n"):
                if not line:
                    continue

                # Ripgrep format: file_path:line_number:content
                parts = line.split(":", 2)
                if len(parts) < 3:
                    if not json_output:
                        console.print(f"[yellow]⚠[/yellow] Skipping malformed ripgrep output (expected file:line:content): {line[:80]}...")
                    continue

                file_path, line_num_str, content = parts

                # Validate line number is actually a number
                if not line_num_str.isdigit():
                    if not json_output:
                        console.print(f"[yellow]⚠[/yellow] Invalid line number '{line_num_str[:20]}' in {file_path}")
                    continue

                if file_path not in matches_by_file:
                    matches_by_file[file_path] = []
                matches_by_file[file_path].append({
                    "line": int(line_num_str),
                    "content": content,
                })

            if json_output:
                # Flatten for JSON output
                results = []
                for fp, matches in matches_by_file.items():
                    for m in matches:
                        results.append({
                            "path": fp,
                            "line": m["line"],
                            "content": m["content"],
                        })
                format_json_output(results, raw=True)
            else:
                format_exact_results(matches_by_file, pattern)

        elif result.returncode == 1:
            # No matches found
            elapsed_ms = int((time.time() - start_time) * 1000)
            if json_output:
                format_json_output({"matches": [], "time_ms": elapsed_ms}, raw=True)
            else:
                # Generate helpful suggestions
                suggestions = []
                if len(pattern) > 10:
                    suggestions.append(f"grepl exact {repr(pattern[:8])} -p {path}")
                if not ignore_case:
                    suggestions.append(f"grepl exact {repr(pattern)} -i -p {path}")
                suggestions.append(f"grepl read {path}")

                format_no_results(
                    pattern=pattern,
                    search_type="EXACT",
                    time_ms=elapsed_ms,
                    suggestions=suggestions,
                )
            sys.exit(ExitCode.NO_MATCHES)
        else:
            raise FileNotFoundError()

    except FileNotFoundError:
        # Fallback to grep
        cmd = ["grep", "-rn", pattern, path]
        if ignore_case:
            cmd.insert(2, "-i")
        if limit:
            cmd.insert(2, f"-m{limit}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            output = result.stdout
            if limit:
                lines = output.strip().split("\n")
                output = "\n".join(lines[:limit])

            # Parse for tree output
            matches_by_file: Dict[str, List[dict]] = {}
            for line in output.strip().split("\n"):
                if ":" not in line:
                    continue
                file_path, line_num, *content_parts = line.split(":", 2)
                content = content_parts[0] if content_parts else ""

                if file_path not in matches_by_file:
                    matches_by_file[file_path] = []
                matches_by_file[file_path].append({
                    "line": int(line_num) if line_num.isdigit() else 0,
                    "content": content,
                })

            if json_output:
                results = []
                for fp, matches in matches_by_file.items():
                    for m in matches:
                        results.append({
                            "path": fp,
                            "line": m["line"],
                            "content": m["content"],
                        })
                format_json_output(results, raw=True)
            else:
                format_exact_results(matches_by_file, pattern)
        else:
            # No matches found (grep fallback)
            elapsed_ms = int((time.time() - start_time) * 1000)
            if json_output:
                format_json_output({"matches": [], "time_ms": elapsed_ms}, raw=True)
            else:
                suggestions = []
                if len(pattern) > 10:
                    suggestions.append(f"grepl exact {repr(pattern[:8])} -p {path}")
                if not ignore_case:
                    suggestions.append(f"grepl exact {repr(pattern)} -i -p {path}")
                suggestions.append(f"grepl read {path}")

                format_no_results(
                    pattern=pattern,
                    search_type="EXACT",
                    time_ms=elapsed_ms,
                    suggestions=suggestions,
                )
            sys.exit(ExitCode.NO_MATCHES)


@main.command()
@click.argument("location")
@click.option("--context", "-c", default=50, help="Lines of context (default: 50)")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
def read(location: str, context: int, json_output: bool):
    """Read file contents with context around a line.

    Usage:
        grepl read src/auth.py              # Read first 50 lines
        grepl read src/auth.py:45           # Read ~50 lines centered on line 45
        grepl read src/auth.py:30-80        # Read lines 30-80
        grepl read src/auth.py -c 100       # More context
    """
    # Parse location: file.py, file.py:line, or file.py:start-end
    if ":" in location:
        file_part, line_part = location.rsplit(":", 1)
        if "-" in line_part:
            # Range: file.py:30-80
            start_str, end_str = line_part.split("-", 1)
            try:
                start_line = int(start_str)
                end_line = int(end_str)
            except ValueError:
                format_error_rich(
                    f"Invalid line range: '{line_part}'",
                    context=f"Expected format: start-end (e.g., 30-80)",
                    why=["Both start and end must be integers"],
                    fixes=[
                        f"grepl read {file_part}:1-50",
                        f"grepl read {file_part}",
                    ],
                )
                sys.exit(ExitCode.PATTERN_ERROR)
        else:
            # Single line: file.py:45
            try:
                center_line = int(line_part)
                half = context // 2
                start_line = max(1, center_line - half)
                end_line = center_line + half
            except ValueError:
                # Maybe it's part of the path (e.g., C:\path on Windows)
                file_part = location
                start_line = 1
                end_line = context
    else:
        file_part = location
        start_line = 1
        end_line = context

    file_path = Path(file_part)
    if not file_path.exists():
        format_error_rich(
            f"File not found: {file_part}",
            context=f"Resolved path: {file_path.resolve()}",
            why=["File path may be misspelled", "File may have been moved or deleted"],
            fixes=[
                f"ls {file_part.rsplit('/', 1)[0] if '/' in file_part else '.'}",
                "grepl exact '<pattern>' -p .",
            ],
            tip="Use grepl exact to search for files",
        )
        sys.exit(ExitCode.PATH_ERROR)

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except PermissionError:
        format_error_rich(
            f"Permission denied: {file_path}",
            context="Cannot read file due to permissions",
            fixes=[f"ls -la {file_path}", f"chmod +r {file_path}"],
        )
        sys.exit(ExitCode.PERMISSION_ERROR)
    except Exception as e:
        format_error_rich(
            f"Error reading file",
            context=str(e),
            fixes=[f"file {file_path}", f"ls -la {file_path}"],
        )
        sys.exit(ExitCode.PATH_ERROR)

    total_lines = len(lines)

    # Clamp to file bounds
    start_line = max(1, start_line)
    end_line = min(total_lines, end_line)

    # Prepare line data
    line_data = []
    for i in range(start_line - 1, end_line):
        line_num = i + 1
        line_content = lines[i].rstrip("\n\r")
        line_data.append({
            "num": line_num,
            "content": line_content
        })

    if json_output:
        json_data = {
            "path": str(file_path),
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": total_lines,
            "lines": line_data
        }
        format_json_output(json_data, raw=True)
        return

    format_read_output(
        str(file_path),
        line_data,
        start_line,
        end_line,
        total_lines
    )


def find_block_bounds(lines: List[str], target_line: int) -> tuple:
    """
    Find the function/class block containing the target line.

    Returns (start_line, end_line, block_type, block_name) or None if no block found.
    Lines are 1-indexed.
    """
    if target_line < 1 or target_line > len(lines):
        return None

    target_idx = target_line - 1
    target_content = lines[target_idx]

    # Get indentation of target line
    target_indent = len(target_content) - len(target_content.lstrip())

    # Pattern to match function/class definitions
    def_pattern = re.compile(r'^(\s*)(def|class|async def)\s+(\w+)')

    # Search backwards for the containing definition
    block_start = None
    block_type = None
    block_name = None
    block_indent = None

    for i in range(target_idx, -1, -1):
        line = lines[i]
        match = def_pattern.match(line)
        if match:
            indent = len(match.group(1))
            # This definition contains our target if its indentation is less than
            # or equal to target's indentation (allowing for the definition line itself)
            if indent <= target_indent or i == target_idx:
                block_start = i + 1  # 1-indexed
                block_type = "function" if "def" in match.group(2) else "class"
                block_name = match.group(3)
                block_indent = indent
                break

    if block_start is None:
        # No containing block found, return file-level context
        return None

    # Search forwards for the end of the block
    block_end = len(lines)  # Default to end of file

    for i in range(target_idx + 1, len(lines)):
        line = lines[i]
        stripped = line.lstrip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            continue

        current_indent = len(line) - len(stripped)

        # Block ends when we hit a line at same or lesser indentation
        # that isn't a continuation of the block
        if current_indent <= block_indent:
            # Check if it's a new definition or other statement at module level
            block_end = i  # Don't include this line
            break

    return (block_start, block_end, block_type, block_name)


@main.command()
@click.argument("location")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
def context(location: str, json_output: bool):
    """Show the function/class containing a target line.

    Usage:
        grepl context src/auth.py:45           # Show function containing line 45
        grepl context src/models.py:120        # Show class/function at line 120
    """
    # Parse location: file.py:line
    if ":" not in location:
        format_error_rich(
            "Invalid format",
            context=f"Got '{location}' but expected 'file.py:line'",
            why=["Missing colon separator between file path and line number"],
            fixes=[
                f"grepl context {location}:50",
                "grepl context src/main.py:42",
            ],
            tip="Format: grepl context <file>:<line>",
        )
        sys.exit(ExitCode.PATTERN_ERROR)

    file_part, line_part = location.rsplit(":", 1)
    try:
        target_line = int(line_part)
    except ValueError:
        format_error_rich(
            f"Invalid line number: '{line_part}'",
            context=f"Tried to parse '{line_part}' as a line number",
            why=["Line number must be a positive integer"],
            fixes=[
                f"grepl context {file_part}:1",
                f"grepl read {file_part}",
            ],
        )
        sys.exit(ExitCode.PATTERN_ERROR)

    file_path = Path(file_part).resolve()

    if not file_path.exists():
        format_error_rich(
            f"File not found: {file_part}",
            context=f"Resolved path: {file_path}",
            why=["File path may be misspelled", "File may have been moved or deleted"],
            fixes=[
                f"ls {file_part.rsplit('/', 1)[0] if '/' in file_part else '.'}",
                "grepl exact '<pattern>' -p .",
            ],
            tip="Use grepl exact to find files containing specific patterns",
        )
        sys.exit(ExitCode.PATH_ERROR)

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except PermissionError:
        format_error_rich(
            f"Permission denied: {file_path}",
            context="Cannot read file due to permissions",
            fixes=[f"ls -la {file_path}", f"chmod +r {file_path}"],
        )
        sys.exit(ExitCode.PERMISSION_ERROR)
    except Exception as e:
        format_error_rich(
            f"Cannot read file",
            context=str(e),
            fixes=[f"file {file_path}", f"ls -la {file_path}"],
        )
        sys.exit(ExitCode.PATH_ERROR)

    total_lines = len(lines)

    if target_line < 1 or target_line > total_lines:
        format_error_rich(
            f"Line {target_line} out of range",
            context=f"File '{file_part}' has {total_lines} lines (1-{total_lines})",
            why=[f"Requested line {target_line} but file only has {total_lines} lines"],
            fixes=[
                f"grepl context {file_part}:{min(target_line, total_lines)}",
                f"grepl read {file_part}",
            ],
            tip=f"Valid line range: 1-{total_lines}",
        )
        sys.exit(ExitCode.PATTERN_ERROR)

    # Find the containing block
    result = find_block_bounds(lines, target_line)

    if result is None:
        # No block found - show context around the line
        half = 25
        start_line = max(1, target_line - half)
        end_line = min(total_lines, target_line + half)
        block_type = "file"
        block_name = ""
    else:
        start_line, end_line, block_type, block_name = result

    # Build line data
    line_data = []
    for i in range(start_line - 1, end_line):
        line_num = i + 1
        line_content = lines[i].rstrip("\n\r")
        line_data.append({
            "num": line_num,
            "content": line_content
        })

    if json_output:
        json_data = {
            "path": str(file_path),
            "target_line": target_line,
            "start_line": start_line,
            "end_line": end_line,
            "block_type": block_type,
            "block_name": block_name,
            "lines": line_data
        }
        format_json_output(json_data, raw=True)
        return

    format_context_output(
        str(file_path),
        line_data,
        start_line,
        end_line,
        target_line,
        block_type,
        block_name
    )


@main.command()
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("-a", "--all", "show_hidden", is_flag=True, help="Show hidden files")
@click.option("-t", "--dirs-first", "dirs_first", is_flag=True, help="Sort directories first")
@click.option("-f", "--flat", "flat", is_flag=True, help="Show only top level (non-recursive)")
@click.option("-d", "--depth", "depth", default=10, help="Max tree depth")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format for agents")
def ls(path: str, show_hidden: bool, dirs_first: bool, flat: bool, depth: int, json_output: bool):
    """List directory contents with tree structure and rich formatting.

    Shows files and directories with relative paths, avoiding system noise.
    Directories are shown with a trailing slash (/) and highlighted in cyan.

    By default shows recursive directory tree (directories only).
    Use -f/--flat for single-level listing with files.
    """
    target_path = Path(path).resolve()

    if not target_path.is_dir():
        format_error_rich(
            f"Not a directory: {path}",
            context=f"Path exists but is not a directory: {target_path}",
            fixes=[f"file {path}", f"grepl read {path}"],
        )
        sys.exit(ExitCode.PATH_ERROR)

    def collect_items(dir_path: Path, current_depth: int) -> List[dict]:
        items = []
        try:
            for entry in dir_path.iterdir():
                is_dir = entry.is_dir()
                if not flat and not is_dir:
                    continue
                item = {
                    "name": entry.name,
                    "is_dir": is_dir,
                }
                if is_dir and not flat and current_depth < depth:
                    item["children"] = collect_items(entry, current_depth + 1)
                items.append(item)
        except PermissionError:
            pass
        return items

    items = collect_items(target_path, 0)

    if json_output:
        json_data = {
            "path": str(target_path),
            "items": items,
        }
        format_json_output(json_data, raw=True)
        return

    try:
        rel_path = target_path.relative_to(Path.cwd())
    except ValueError:
        rel_path = target_path
    format_ls_output(str(rel_path), items, show_hidden, dirs_first, not flat)


@main.command()
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format for agents")
def status(path: str, json_output: bool):
    """Check indexing status.

    Returns comprehensive index state for both humans and agents.
    Use --json for machine-readable output suitable for automation.
    """
    project_path = Path(path).resolve()
    stats = get_detailed_stats(project_path)

    if json_output:
        print(json.dumps(stats, indent=2))
        return

    format_status_output(
        project_path=str(project_path),
        indexed=stats["indexed"],
        chunks=stats.get("chunks", 0),
        files=stats.get("files", 0),
        last_updated=stats.get("lastIndexedAt"),
        ollama_running=stats.get("ollamaRunning", False),
        model_available=stats.get("modelAvailable", False),
        semantic_ready=stats.get("semanticReady", False),
        semantic_reason=stats.get("semanticReadyReason"),
    )


@main.group(invoke_without_command=True)
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
@click.pass_context
def model(ctx, json_output: bool):
    """Manage embedding model backends.

    Run without arguments to show current backend and available options.
    Use 'grepl model use <backend>' to switch backends.
    """
    if ctx.invoked_subcommand is None:
        backend_info = get_backend_info()
        available_backends = list_available_backends()

        if json_output:
            print(json.dumps({
                "current": backend_info,
                "available": available_backends,
            }, indent=2))
            return

        label = badge("MODEL", Colors.BRIGHT_CYAN)
        print(f"\n{label} {dim('Embedding Backend')}\n")

        current_backend = backend_info["backend"]
        current_model = backend_info["model"]
        is_available = backend_info["available"]

        status_icon = green("✓") if is_available else yellow("○")
        print(f"  {dim('Current:')} {status_icon} {green(current_backend)} {dim('─')} {cyan(current_model)}")
        print(f"  {dim('Dimensions:')} {backend_info['dimensions']}")

        if current_backend == "ollama" and "url" in backend_info:
            print(f"  {dim('URL:')} {backend_info['url']}")

        print(f"\n  {dim('Available Backends:')}\n")

        for be in available_backends:
            icon = green("✓") if be["available"] else dim("○")
            name_display = green(be["name"]) if be["available"] else dim(be["name"])
            model_display = cyan(be["model"]) if be["available"] else dim(be["model"])
            reason_display = dim(f"({be['reason']})")

            prefix = "├──" if be != available_backends[-1] else "└──"
            print(f"  {dim(prefix)} {icon} {name_display:8} {dim('─')} {model_display:30} {reason_display}")

        print(f"\n  {dim('Switch backend:')} {cyan('grepl model use <backend>')}")
        print(f"  {dim('Example:')} {cyan('grepl model use ollama')}\n")


@model.command(name="use")
@click.argument("backend", type=click.Choice(["openai", "ollama"], case_sensitive=False))
def model_use(backend: str):
    """Switch to a different embedding backend."""
    backend = backend.lower()

    available_backends = list_available_backends()
    backend_obj = next((b for b in available_backends if b["name"] == backend), None)

    if not backend_obj:
        print(format_error(f"Unknown backend: {backend}"))
        sys.exit(1)

    if not backend_obj["available"]:
        print(format_error(f"Backend '{backend}' is not available"))
        print(f"  {dim('Reason:')} {backend_obj['reason']}")
        sys.exit(1)

    set_preferred_backend(backend)

    label = badge("MODEL", Colors.BRIGHT_CYAN)
    print(f"{label} {green('Switched to')} {cyan(backend)} {dim('─')} {dim(backend_obj['model'])}")
    print(f"\n  {yellow('Note:')} {dim('You may need to reindex with')} {cyan('grepl index . --force')} {dim('if switching between backends')}\n")


@main.command()
@click.argument("path", default=".", type=click.Path(exists=True))
def clear(path: str):
    """Clear the search index."""
    project_path = Path(path).resolve()
    clear_index(project_path)

    label = badge("CLEAR", Colors.BRIGHT_YELLOW)
    print(f"{label} {green('Index cleared for')} {cyan(str(project_path))}")


if __name__ == "__main__":
    main()
