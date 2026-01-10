"""Output formatting utilities for grepl CLI."""

import json
from pathlib import Path
from typing import Optional

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import get_lexer_by_name, get_lexer_for_filename
from pygments.util import ClassNotFound

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text


console = Console()


def get_lexer_for_file(file_path: str):
    """Get a Pygments lexer for a file based on its extension."""
    try:
        return get_lexer_for_filename(file_path)
    except ClassNotFound:
        # Default to a generic text lexer
        try:
            return get_lexer_by_name('text')
        except ClassNotFound:
            return None


def syntax_highlight_code(code: str, file_path: str) -> str:
    """
    Apply Pygments syntax highlighting to code.

    Args:
        code: The source code to highlight
        file_path: Path to the file (for lexer detection)

    Returns:
        Highlighted code string
    """
    lexer = get_lexer_for_file(file_path)
    if not lexer:
        return code

    formatter = Terminal256Formatter(style='monokai')
    return highlight(code, lexer, formatter)


def truncate_line(line: str, max_length: int = 120):
    """
    Truncate a line if it exceeds max_length, with an indicator.

    Args:
        line: The line to potentially truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated line with indicator, or original line
    """
    if len(line) <= max_length:
        return line
    return line[:max_length - 3] + '...'


def format_score_color(score: float) -> str:
    """
    Get color code for search result score.

    Args:
        score: Similarity score (0-1)

    Returns:
        Rich color string (green, yellow, or red)
    """
    if score >= 0.8:
        return '[green]'
    elif score >= 0.5:
        return '[yellow]'
    else:
        return '[red]'


def format_json_output(data, raw: bool = False):
    """
    Format output as JSON.

    Args:
        data: Data to serialize
        raw: If True, print JSON; if False, return formatted string

    Returns:
        JSON string if raw=False, otherwise None
    """
    import os
    import sys

    json_str = json.dumps(data, indent=2)

    if raw:
        # Use Rich syntax highlighting for TTY, plain JSON when piped
        if sys.stdout.isatty() and not os.environ.get("NO_COLOR"):
            console.print_json(json_str)
        else:
            print(json_str)
    else:
        return json_str


def create_file_header(file_path: str) -> str:
    """
    Create a styled file header.

    Args:
        file_path: Path to format

    Returns:
        Formatted header string
    """
    path = Path(file_path)
    short_path = file_path
    if path.exists():
        try:
            short_path = f"[cyan]{file_path}[/cyan]"
        except:
            short_path = file_path
    return f"[bold]{short_path}[/bold]"


def format_search_result(result: dict, show_multiline: bool = True, max_lines: int = 3) -> None:
    """
    Format and print a single search result.

    Args:
        result: Search result dict with keys: file_path, start_line, end_line, content, score
        show_multiline: Whether to show multiple lines of content
        max_lines: Maximum number of lines to show
    """
    file_path = result['file_path']
    start_line = result['start_line']
    content = result['content']
    score = result['score']

    # Split content into lines
    lines = content.split('\n')
    
    # Truncate to max_lines
    if show_multiline:
        lines = lines[:max_lines]
    else:
        lines = lines[:1]

    # Format location header
    location = f"[cyan]{file_path}:{start_line}[/cyan]"
    if result.get('end_line') and result['end_line'] != start_line:
        location = f"[cyan]{file_path}:{start_line}-{result['end_line']}[/cyan]"

    # Format score
    score_color = format_score_color(score)
    score_text = f"{score_color}(score: {score:.2f})[/]"

    print(f"{location} {score_text}")

    # Print content with syntax highlighting
    for line in lines:
        highlighted = syntax_highlight_code(line, file_path)
        print(f"    {highlighted}")

    # Add truncation indicator
    if len(content.split('\n')) > max_lines:
        print("    [dim]...[/dim]")


def grouped_output(matches: list[str], file_path: Optional[str] = None) -> None:
    """
    Print grouped output with Rich Panel.

    Args:
        matches: List of match strings
        file_path: Optional file path for header
    """
    if file_path:
        header = create_file_header(file_path)
        console.print(Panel("\n".join(matches), title=header, border_style="blue"))
    else:
        console.print("\n".join(matches))
