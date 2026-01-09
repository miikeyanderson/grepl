"""Grepl CLI - Semantic code search."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .embedder import check_ollama, check_model
from .chunker import chunk_codebase
from .store import index_chunks, search, clear_index, get_stats, has_index
from .utils.formatters import format_json_output
from .utils.tree_formatter import (
    format_exact_results,
    format_search_results,
    format_read_output,
    format_status_output,
    format_index_header,
    format_index_progress,
    format_error,
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


@main.command()
@click.argument("pattern")
@click.option("--limit", "-n", default=None, type=int, help="Max number of results")
@click.option("--ignore-case", "-i", is_flag=True, help="Case-insensitive search")
@click.option("--path", "-p", default=".", help="Path to search")
@click.option("--json", "json_output", is_flag=True, help="Output in JSON format")
def exact(pattern: str, limit: int, ignore_case: bool, path: str, json_output: bool):
    """Exact pattern search (uses ripgrep/grep)."""
    try:
        # Try ripgrep first
        cmd = ["rg", "-n", "--color=never", pattern, path]
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
            if json_output:
                format_json_output([], raw=True)
            else:
                label = badge("EXACT", Colors.BRIGHT_MAGENTA)
                print(f"{label} {cyan(repr(pattern))} {dim('──')} {dim('0 matches')}")
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
            if json_output:
                format_json_output([], raw=True)
            else:
                label = badge("EXACT", Colors.BRIGHT_MAGENTA)
                print(f"{label} {cyan(repr(pattern))} {dim('──')} {dim('0 matches')}")


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
                format_error(f"Invalid line range: {line_part}")
                sys.exit(1)
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
        format_error(f"File not found: {file_part}")
        sys.exit(1)

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        format_error(f"Error reading file: {e}")
        sys.exit(1)

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
