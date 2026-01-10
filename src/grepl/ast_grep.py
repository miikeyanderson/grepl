"""AST-grep integration for structural code search."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .ranker import Hit


# Rule search paths (in order of precedence)
def _get_rule_search_paths(project_path: Optional[Path] = None) -> List[Path]:
    """Get rule search paths in order of precedence."""
    paths = []

    # 1. Project-local rules (highest precedence)
    if project_path:
        paths.append(project_path / ".grepl" / "rules")

    # 2. User rules
    user_rules = Path.home() / ".grepl" / "rules"
    paths.append(user_rules)

    # 3. Packaged rules (shipped with grepl)
    package_rules = Path(__file__).parent / "rules"
    paths.append(package_rules)

    return paths


def resolve_rule_file(
    rule_name: str,
    project_path: Optional[Path] = None,
) -> Tuple[Optional[Path], str]:
    """
    Resolve a rule name to a file path.

    Args:
        rule_name: Either a path to a rule file or a rule name (without .yml)
        project_path: Project root for local rule resolution

    Returns:
        Tuple of (resolved_path, source) where source is "path", "project", "user", or "builtin"
    """
    # If it's already a path, use it directly
    rule_path = Path(rule_name)
    if rule_path.suffix in ('.yml', '.yaml') and rule_path.exists():
        return rule_path, "path"

    # Search in rule directories
    search_paths = _get_rule_search_paths(project_path)
    sources = ["project", "user", "builtin"]

    for search_path, source in zip(search_paths, sources):
        if not search_path.exists():
            continue

        # Try with .yml extension
        candidate = search_path / f"{rule_name}.yml"
        if candidate.exists():
            return candidate, source

        # Try with .yaml extension
        candidate = search_path / f"{rule_name}.yaml"
        if candidate.exists():
            return candidate, source

    return None, "not_found"


def list_available_rules(project_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """List available rules by source."""
    rules: Dict[str, List[str]] = {
        "project": [],
        "user": [],
        "builtin": [],
    }

    search_paths = _get_rule_search_paths(project_path)
    sources = ["project", "user", "builtin"]

    for search_path, source in zip(search_paths, sources):
        if not search_path.exists():
            continue

        for rule_file in search_path.glob("*.yml"):
            rules[source].append(rule_file.stem)
        for rule_file in search_path.glob("*.yaml"):
            rules[source].append(rule_file.stem)

    return rules


@dataclass
class AstGrepConfig:
    """Configuration for ast-grep execution."""
    patterns: List[str]
    rule_files: List[str]
    language: Optional[str] = None
    exhaustive: bool = False


def check_ast_grep() -> bool:
    """Check if ast-grep (sg) is installed."""
    return shutil.which("sg") is not None


def get_ast_grep_version() -> Optional[str]:
    """Get ast-grep version if installed."""
    if not check_ast_grep():
        return None
    try:
        result = subprocess.run(
            ["sg", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def format_ast_grep_install_hint() -> str:
    """Return helpful install instructions for ast-grep."""
    return """ast-grep (sg) is not installed.

Install with:
  brew install ast-grep    # macOS
  cargo install ast-grep   # Cross-platform (requires Rust)
  npm install -g @ast-grep/cli  # Via npm

Or visit: https://ast-grep.github.io/guide/quick-start.html

To run without AST stage, remove --ast flags."""


def _infer_language(file_path: str) -> Optional[str]:
    """Infer language from file extension for ast-grep."""
    ext_map = {
        ".swift": "swift",
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".kt": "kotlin",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".lua": "lua",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    ext = Path(file_path).suffix.lower()
    return ext_map.get(ext)


def run_ast_grep(
    pattern: str,
    files: Optional[List[str]] = None,
    root_path: Optional[Path] = None,
    language: Optional[str] = None,
) -> List[Hit]:
    """
    Run ast-grep with a pattern and return hits.

    Args:
        pattern: AST pattern to search for (e.g., "try { $$ } catch { $$ }")
        files: Optional list of files to search (narrows scope)
        root_path: Root path to search if no files specified
        language: Language hint (e.g., "swift", "python")

    Returns:
        List of Hit objects from ast-grep matches
    """
    if not check_ast_grep():
        return []

    hits: List[Hit] = []

    # Build command
    cmd = ["sg", "--pattern", pattern, "--json"]

    if language:
        cmd.extend(["--lang", language])

    # Determine what to search
    if files:
        # Search specific files
        for file_path in files:
            file_lang = language or _infer_language(file_path)
            file_cmd = ["sg", "--pattern", pattern, "--json"]
            if file_lang:
                file_cmd.extend(["--lang", file_lang])
            file_cmd.append(file_path)

            try:
                result = subprocess.run(
                    file_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    hits.extend(_parse_ast_grep_output(result.stdout, pattern))
            except subprocess.TimeoutExpired:
                continue
            except Exception:
                continue
    else:
        # Search from root path
        search_path = str(root_path) if root_path else "."
        cmd.append(search_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0 and result.stdout.strip():
                hits.extend(_parse_ast_grep_output(result.stdout, pattern))
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

    return hits


def run_ast_grep_rule(
    rule_file: str,
    files: Optional[List[str]] = None,
    root_path: Optional[Path] = None,
) -> List[Hit]:
    """
    Run ast-grep with a rule file and return hits.

    Args:
        rule_file: Path to YAML rule file
        files: Optional list of files to search
        root_path: Root path to search if no files specified

    Returns:
        List of Hit objects from ast-grep matches
    """
    if not check_ast_grep():
        return []

    hits: List[Hit] = []

    # Build command for rule file
    cmd = ["sg", "scan", "--rule", rule_file, "--json"]

    if files:
        # For rule files, we need to search each file
        for file_path in files:
            file_cmd = ["sg", "scan", "--rule", rule_file, "--json", file_path]
            try:
                result = subprocess.run(
                    file_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    hits.extend(_parse_ast_grep_output(result.stdout, rule_file=rule_file))
            except Exception:
                continue
    else:
        search_path = str(root_path) if root_path else "."
        cmd.append(search_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0 and result.stdout.strip():
                hits.extend(_parse_ast_grep_output(result.stdout, rule_file=rule_file))
        except Exception:
            pass

    return hits


def _read_file_snippet(file_path: str, start_line: int, end_line: int, context: int = 2) -> str:
    """Read a snippet from a file with context lines."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # Adjust for 0-based indexing and add context
        start_idx = max(0, start_line - 1 - context)
        end_idx = min(len(lines), end_line + context)

        snippet_lines = lines[start_idx:end_idx]
        return ''.join(snippet_lines).rstrip()
    except Exception:
        return ""


def _parse_ast_grep_output(
    output: str,
    pattern: Optional[str] = None,
    rule_file: Optional[str] = None,
) -> List[Hit]:
    """Parse ast-grep JSON output into Hit objects."""
    hits: List[Hit] = []

    try:
        # ast-grep outputs a JSON array (not NDJSON)
        parsed = json.loads(output.strip())
        if not isinstance(parsed, list):
            parsed = [parsed]

        for match in parsed:
            # ast-grep JSON format has: file, range, text, lines, etc.
            file_path = match.get("file", "")
            if not file_path:
                continue

            # Range contains start/end with line/column (0-indexed)
            range_info = match.get("range", {})
            start = range_info.get("start", {})
            end = range_info.get("end", {})

            # sg uses 0-indexed lines, convert to 1-indexed
            start_line = start.get("line", 0) + 1
            end_line = end.get("line", start.get("line", 0)) + 1
            start_col = start.get("column")
            end_col = end.get("column")

            # Use "lines" field if available (has better context), fallback to "text"
            preview = match.get("lines", "") or match.get("text", "")
            if preview and len(preview) < 50:
                # Short match - read file snippet for better context
                file_snippet = _read_file_snippet(file_path, start_line, end_line, context=1)
                if file_snippet:
                    preview = file_snippet

            # Extract any captured metavariables
            # New structure: metaVariables: {single: {}, multi: {}, transformed: {}}
            captures: Dict[str, str] = {}
            meta_vars = match.get("metaVariables", {})
            # Handle single metavariables
            for var_name, var_info in meta_vars.get("single", {}).items():
                if isinstance(var_info, dict):
                    captures[var_name] = var_info.get("text", "")
                elif isinstance(var_info, str):
                    captures[var_name] = var_info
            # Handle multi metavariables (like $$$ARGS)
            for var_name, var_list in meta_vars.get("multi", {}).items():
                if isinstance(var_list, list) and var_list:
                    # Join all captured items
                    texts = [v.get("text", "") for v in var_list if isinstance(v, dict)]
                    captures[var_name] = ", ".join(texts)

            hit = Hit(
                source="ast",
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                score=0.0,
                preview=preview,
                symbols=[],
                ast_score=0.9,  # High base score for AST matches
                ast_pattern=pattern,
                ast_rule=rule_file,
                ast_captures=captures,
                start_col=start_col,
                end_col=end_col,
            )
            hits.append(hit)

    except Exception:
        pass

    return hits


def run_ast_grep_multi(
    config: AstGrepConfig,
    files: Optional[List[str]] = None,
    root_path: Optional[Path] = None,
) -> List[Hit]:
    """
    Run multiple ast-grep patterns and rules, combining results.

    Args:
        config: AST grep configuration with patterns and rules
        files: Optional list of files to search
        root_path: Root path to search if no files specified

    Returns:
        Combined list of Hit objects
    """
    all_hits: List[Hit] = []

    # Run each pattern
    for pattern in config.patterns:
        hits = run_ast_grep(
            pattern=pattern,
            files=files,
            root_path=root_path,
            language=config.language,
        )
        all_hits.extend(hits)

    # Run each rule file
    for rule_file in config.rule_files:
        hits = run_ast_grep_rule(
            rule_file=rule_file,
            files=files,
            root_path=root_path,
        )
        all_hits.extend(hits)

    return all_hits
