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
from .embedder import check_ollama, check_model
from .chunker import chunk_codebase
from .store import index_chunks, search, clear_index, get_stats, has_index
from .planner import analyze_query
from .ranker import Hit, RankWeights, merge_results, rerank
from .utils.formatters import format_json_output
from .utils.tree_formatter import (
    format_exact_results,
    format_search_results,
    format_find_results,
    format_read_output,
    format_context_output,
    format_status_output,
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


@click.group()
@click.version_option(version=__version__)
def main():
    """Grepl - Semantic code search powered by ChromaDB + Ollama."""
    pass


@main.command()
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("--force", "-f", is_flag=True, help="Force reindex")
def index(path: str, force: bool):
    """Index a codebase for semantic search."""
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

    # Check if already indexed
    if has_index(project_path) and not force:
        stats = get_stats(project_path)
        print(f"  {yellow('!')} Index already exists with {cyan(str(stats['chunks']))} chunks")
        print(f"  {dim('Use')} {cyan('--force')} {dim('to reindex')}")
        return

    if force:
        # Delete and recreate the collection to ensure index settings (e.g. HNSW space) apply.
        format_index_progress("Clearing existing index...")
        clear_index(project_path)
        format_index_progress("Cleared", done=True)

    # Collect chunks
    format_index_progress("Scanning files...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning...", total=None)
        chunks = list(chunk_codebase(project_path))
        progress.update(task, description=f"Found {len(chunks)} chunks")

    format_index_progress(f"Found {len(chunks)} chunks", done=True)

    if not chunks:
        print(f"  {yellow('!')} No files found to index")
        return

    # Index chunks
    format_index_progress("Generating embeddings...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Indexing...", total=None)
        index_chunks(project_path, chunks)
        progress.update(task, description="Done")

    format_index_progress(f"Indexed {len(chunks)} chunks", done=True)
    print()
    print(f"  {green('Ready!')} Run {cyan('grepl search <query>')} to search")


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
@click.argument("query")
@click.option("-k", "--top-k", default=10, show_default=True, help="Number of results")
@click.option("-p", "--path", default=".", help="Search path")
@click.option("--lang", default=None, help="Limit to languages (e.g. py,ts,swift)")
@click.option("--grep-only", is_flag=True, help="Only use grep")
@click.option("--semantic-only", is_flag=True, help="Only use semantic")
@click.option("--precise", is_flag=True, help="Require grep confirmation")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
def find_cmd(
    query: str,
    top_k: int,
    path: str,
    lang: Optional[str],
    grep_only: bool,
    semantic_only: bool,
    precise: bool,
    json_output: bool,
):
    """Hybrid search: combines exact matching (ripgrep) with semantic search (ChromaDB).

    Modes:
      • Auto (default): chooses grep vs semantic vs hybrid based on the query.
      • --grep-only: exact matches only (no indexing required).
      • --semantic-only: semantic matches only (requires: grepl index + Ollama running).

    Examples:
      grepl find "TopNavBar" -p Faithflow/Shared/DesignSystem/Components/Headers
      grepl find "where do we store onboarding completion" -p Faithflow/App
      grepl find "save push token after login" -p Faithflow/System/Notifications

      # Force mode
      grepl find "registerDeviceToken" -p Faithflow/System/Notifications --grep-only
      grepl find "listen for customer info updates" -p Faithflow/System/IAP --semantic-only

      # Notes
      • Semantic search requires an index: grepl index <path>
      • Semantic search requires Ollama: ollama serve (and model: nomic-embed-text)
    """
    start_time = time.time()
    project_path = Path(path).resolve()
    exts = _lang_to_exts(lang)

    plan = analyze_query(query, grep_only=grep_only, semantic_only=semantic_only, precise=precise)
    mode = plan.mode

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

    # Grep
    if plan.run_grep and plan.grep_pattern:
        max_grep = max(50, min(500, top_k * 25))
        matches = _run_rg(plan.grep_pattern, project_path, fixed=plan.grep_fixed, max_results=max_grep, exts=exts)
        grep_hits = _grep_matches_to_hits(matches, grep_score=grep_hit_score)

    # Semantic
    if plan.run_semantic and plan.semantic_query:
        if not check_ollama():
            if semantic_only and not grep_hits:
                if json_output:
                    print(json.dumps({"error": "Ollama is not running"}))
                else:
                    format_error("Ollama is not running", hint=f"Start with: {cyan('ollama serve')}")
                sys.exit(1)
        elif not has_index(project_path):
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

    # Merge + rank
    merged = merge_results(grep_hits, semantic_hits, overlap_lines=3)
    ranked = rerank(merged, weights=RankWeights(), max_per_file=3)

    # Reflect what actually ran/returned.
    effective_mode = mode
    if grep_hits and not semantic_hits:
        effective_mode = "exact"
    elif semantic_hits and not grep_hits:
        effective_mode = "semantic"
    elif semantic_hits and grep_hits:
        effective_mode = "hybrid"

    if precise:
        ranked_precise = [h for h in ranked if h.grep_score > 0]
        if ranked_precise:
            ranked = ranked_precise
            effective_mode = "exact" if not semantic_hits else effective_mode

    ranked = ranked[: max(0, top_k)]

    elapsed_ms = int((time.time() - start_time) * 1000)
    if json_output:
        payload = {
            "query": query,
            "mode": effective_mode,
            "time_ms": elapsed_ms,
            "results": [
                {
                    "source": h.source,
                    "file_path": h.file_path,
                    "start_line": h.start_line,
                    "end_line": h.end_line,
                    "score": h.score,
                    "grep_score": h.grep_score,
                    "semantic_score": h.semantic_score,
                    "preview": h.preview,
                    "symbols": h.symbols,
                }
                for h in ranked
            ],
        }
        format_json_output(payload, raw=True)
        return

    if not ranked:
        suggestions = []
        if plan.grep_pattern:
            suggestions.append(f"grepl exact {repr(plan.grep_pattern)} -p {path}")
        suggestions.append(f"grepl search {repr(query)} -p {path}")
        if not has_index(project_path):
            suggestions.append(f"grepl index {path}")
        format_no_results(pattern=query, search_type="FIND", time_ms=elapsed_ms, suggestions=suggestions)
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
        cmd = ["rg", "-n", "--color=never", pattern, str(search_path)]
        if ignore_case:
            cmd.insert(2, "-i")
        if limit:
            cmd.insert(2, f"--max-count={limit}")
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
                if ":" not in line:
                    continue
                file_path, line_num, *content_parts = line.split(":", 2)
                content = content_parts[0] if content_parts else ""

                if file_path not in matches_by_file:
                    matches_by_file[file_path] = []
                matches_by_file[file_path].append({
                    "line": int(line_num),
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
def status(path: str):
    """Check indexing status."""
    project_path = Path(path).resolve()
    stats = get_stats(project_path)

    format_status_output(
        project_path=str(project_path),
        indexed=stats["exists"],
        chunks=stats.get("chunks", 0),
    )


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
