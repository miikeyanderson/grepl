"""Tree-style output formatting for grepl CLI."""

import sys
from dataclasses import dataclass
from typing import List, Optional, Dict


# Tree drawing characters
TREE_BRANCH = "├──"
TREE_LAST = "└──"
TREE_PIPE = "│"
TREE_SPACE = "   "
TREE_LINE = "──"

# Color codes (ANSI)
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


def supports_color() -> bool:
    """Check if the terminal supports color output.

    Default: colors ON (most terminals support ANSI colors)
    Opt-out: set NO_COLOR=1 to disable (follows no-color.org standard)
    """
    import os
    # Respect NO_COLOR standard (https://no-color.org/)
    if os.environ.get("NO_COLOR"):
        return False
    # Default to color - most modern terminals support it
    return True


def colorize(text: str, color: str, force: bool = False) -> str:
    """Apply color to text if supported."""
    if not force and not supports_color():
        return text
    return f"{color}{text}{Colors.RESET}"


def badge(text: str, color: str = Colors.BRIGHT_CYAN) -> str:
    """Create a colored badge/label."""
    return colorize(f"{text}", color + Colors.BOLD)


def dim(text: str) -> str:
    """Make text dim/muted."""
    return colorize(text, Colors.DIM)


def cyan(text: str) -> str:
    """Cyan colored text."""
    return colorize(text, Colors.CYAN)


def green(text: str) -> str:
    """Green colored text."""
    return colorize(text, Colors.GREEN)


def yellow(text: str) -> str:
    """Yellow colored text."""
    return colorize(text, Colors.YELLOW)


def red(text: str) -> str:
    """Red colored text."""
    return colorize(text, Colors.RED)


def magenta(text: str) -> str:
    """Magenta colored text."""
    return colorize(text, Colors.MAGENTA)


def bold(text: str) -> str:
    """Bold text."""
    return colorize(text, Colors.BOLD)


def score_color(score: float) -> str:
    """Get appropriate color for a similarity score."""
    if score >= 0.8:
        return Colors.GREEN
    elif score >= 0.5:
        return Colors.YELLOW
    return Colors.RED


def format_score(score: float) -> str:
    """Format a score with appropriate color."""
    color = score_color(score)
    return colorize(f"{score:.2f}", color)


@dataclass
class TreeNode:
    """A node in the tree output."""
    content: str
    children: List["TreeNode"] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


def render_tree(nodes: List[TreeNode], prefix: str = "") -> List[str]:
    """Render a list of tree nodes as formatted lines."""
    lines = []
    for i, node in enumerate(nodes):
        is_last = i == len(nodes) - 1
        connector = TREE_LAST if is_last else TREE_BRANCH

        # Render this node
        tree_prefix = dim(connector) if supports_color() else connector
        lines.append(f"{prefix}{tree_prefix} {node.content}")

        # Render children
        if node.children:
            child_prefix = prefix + (TREE_SPACE if is_last else dim(TREE_PIPE) + "  " if supports_color() else TREE_PIPE + "  ")
            lines.extend(render_tree(node.children, child_prefix))

    return lines


def print_tree(nodes: List[TreeNode], prefix: str = "") -> None:
    """Print a tree structure."""
    for line in render_tree(nodes, prefix):
        print(line)


# ============================================================================
# Command-specific formatters
# ============================================================================

def format_exact_header(pattern: str, match_count: int, file_count: int) -> str:
    """Format header for exact search results."""
    label = badge("EXACT", Colors.BRIGHT_MAGENTA)
    pattern_str = cyan(f'"{pattern}"')
    stats = dim(f"{match_count} matches in {file_count} files")
    return f"{label} {pattern_str} {dim(TREE_LINE)} {stats}"


def format_exact_results(matches_by_file: Dict[str, List[dict]], pattern: str) -> None:
    """
    Print exact search results in tree format.

    Args:
        matches_by_file: Dict mapping file paths to list of {line, content} dicts
        pattern: The search pattern (for header)
    """
    # Count totals
    total_matches = sum(len(matches) for matches in matches_by_file.values())
    file_count = len(matches_by_file)

    # Print header
    print(format_exact_header(pattern, total_matches, file_count))
    print()

    # Build tree nodes
    file_nodes = []
    for file_path, matches in matches_by_file.items():
        # File node with match count
        file_label = f"{cyan(file_path)} {dim(f'({len(matches)})')}"

        # Match children
        match_nodes = []
        for match in matches:
            line_num = match.get("line", match.get("line_num", "?"))
            content = match.get("content", "").strip()
            # Truncate long lines
            if len(content) > 80:
                content = content[:77] + "..."
            match_label = f"{yellow(str(line_num))}: {content}"
            match_nodes.append(TreeNode(match_label))

        file_nodes.append(TreeNode(file_label, match_nodes))

    print_tree(file_nodes)


def format_search_header(query: str, result_count: int, best_score: Optional[float] = None) -> str:
    """Format header for semantic search results."""
    label = badge("SEARCH", Colors.BRIGHT_GREEN)
    query_str = cyan(f'"{query}"')
    stats = f"{result_count} results"
    if best_score is not None:
        stats += f" (best: {format_score(best_score)})"
    return f"{label} {query_str} {dim(TREE_LINE)} {dim(stats)}"


def format_search_results(results: List[dict], query: str, max_lines: int = 3) -> None:
    """
    Print semantic search results in tree format.

    Args:
        results: List of search result dicts
        query: The search query (for header)
        max_lines: Max content lines to show per result
    """
    if not results:
        print(format_search_header(query, 0))
        print()
        print(dim("  No results found."))
        return

    best_score = max(r.get("score", 0) for r in results)
    print(format_search_header(query, len(results), best_score))
    print()

    # Build tree nodes
    result_nodes = []
    for r in results:
        file_path = r.get("file_path", "unknown")
        start_line = r.get("start_line", "?")
        end_line = r.get("end_line", start_line)
        score = r.get("score", 0)
        content = r.get("content", "")

        # Location string
        if start_line != end_line:
            loc = f"{file_path}:{start_line}-{end_line}"
        else:
            loc = f"{file_path}:{start_line}"

        # Node label with score
        label = f"{cyan(loc)} {dim(TREE_LINE)} {format_score(score)}"

        # Content as children (first few lines)
        content_nodes = []
        lines = content.split("\n")[:max_lines]
        for line in lines:
            line = line.rstrip()
            if len(line) > 80:
                line = line[:77] + "..."
            if line.strip():
                content_nodes.append(TreeNode(dim(line)))

        if len(content.split("\n")) > max_lines:
            content_nodes.append(TreeNode(dim("...")))

        result_nodes.append(TreeNode(label, content_nodes))

    print_tree(result_nodes)


def format_read_header(file_path: str, start_line: int, end_line: int, total_lines: int) -> str:
    """Format header for file read output."""
    label = badge("READ", Colors.BRIGHT_BLUE)
    path_str = cyan(file_path)
    range_str = dim(f"lines {start_line}-{end_line} of {total_lines}")
    return f"{label} {path_str} {dim(TREE_LINE)} {range_str}"


def format_read_output(
    file_path: str,
    lines: List[dict],
    start_line: int,
    end_line: int,
    total_lines: int
) -> None:
    """
    Print file content in tree-style format.

    Args:
        file_path: Path to the file
        lines: List of {num, content} dicts
        start_line: First line number
        end_line: Last line number
        total_lines: Total lines in file
    """
    print(format_read_header(file_path, start_line, end_line, total_lines))
    print()

    # Calculate line number width
    width = len(str(end_line))

    for line_data in lines:
        num = line_data.get("num", "?")
        content = line_data.get("content", "")

        # Line number (right-aligned, dimmed)
        line_num = dim(f"{num:>{width}}")

        # Vertical separator
        sep = dim("│")

        print(f"  {line_num} {sep} {content}")


def format_status_header(project_path: str) -> str:
    """Format header for status output."""
    label = badge("STATUS", Colors.BRIGHT_YELLOW)
    path_str = cyan(str(project_path))
    return f"{label} {path_str}"


def format_status_output(
    project_path: str,
    indexed: bool,
    chunks: int = 0,
    files: int = 0,
    last_updated: Optional[str] = None
) -> None:
    """
    Print index status in tree format.

    Args:
        project_path: Path to the project
        indexed: Whether the project is indexed
        chunks: Number of indexed chunks
        files: Number of indexed files
        last_updated: Human-readable time since last update
    """
    print(format_status_header(project_path))
    print()

    status_nodes = []

    if indexed:
        status_nodes.append(TreeNode(f"Indexed: {green('Yes')}"))
        status_nodes.append(TreeNode(f"Chunks: {cyan(str(chunks))}"))
        if files > 0:
            status_nodes.append(TreeNode(f"Files: {cyan(str(files))}"))
        if last_updated:
            status_nodes.append(TreeNode(f"Updated: {dim(last_updated)}"))
    else:
        status_nodes.append(TreeNode(f"Indexed: {yellow('No')}"))
        status_nodes.append(TreeNode(f"Run: {cyan('grepl index .')} to index"))

    print_tree(status_nodes)


def format_index_header(project_path: str) -> str:
    """Format header for indexing output."""
    label = badge("INDEX", Colors.BRIGHT_CYAN)
    path_str = cyan(str(project_path))
    return f"{label} {path_str}"


def format_index_progress(message: str, done: bool = False) -> None:
    """Print an indexing progress message."""
    prefix = green("✓") if done else yellow("○")
    print(f"  {prefix} {message}")


def format_error(message: str, hint: Optional[str] = None) -> None:
    """Print an error message in tree style."""
    label = badge("ERROR", Colors.BRIGHT_RED)
    print(f"{label} {red(message)}")
    if hint:
        print()
        print(f"  {dim(TREE_LAST)} {dim('Hint:')} {hint}")
