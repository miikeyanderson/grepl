"""Code chunking - splits files into searchable chunks."""

import ast
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Generator, Optional

from dataclasses import field

# File extensions to index
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".rs", ".java", ".kt",
    ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".swift",
    ".md", ".txt", ".yaml", ".yml", ".json",
}

# Directories to skip
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv", "myenv", "env",
    "dist", "build", ".next", ".nuxt", "target",
    ".idea", ".vscode", "vendor", ".cache",
    "data", "research_data", "research_data2",  # Large data directories
}

# Max chunk size in characters
MAX_CHUNK_SIZE = 2000
OVERLAP_SIZE = 200


@dataclass
class CodeChunk:
    """A chunk of code for indexing."""
    file_path: str
    start_line: int
    end_line: int
    content: str
    chunk_hash: str
    symbols: List[str] = field(default_factory=list)
    language: str = ""
    last_modified: float = 0.0

    @property
    def id(self) -> str:
        return self.chunk_hash


def hash_content(content: str) -> str:
    """Generate hash for content."""
    return hashlib.md5(content.encode()).hexdigest()[:12]


def _language_for_path(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext in {".ts", ".tsx"}:
        return "typescript"
    if ext in {".js", ".jsx"}:
        return "javascript"
    if ext == ".swift":
        return "swift"
    if ext == ".go":
        return "go"
    if ext == ".rs":
        return "rust"
    if ext == ".rb":
        return "ruby"
    if ext == ".java":
        return "java"
    if ext == ".kt":
        return "kotlin"
    if ext in {".c", ".h"}:
        return "c"
    if ext in {".cpp", ".hpp"}:
        return "cpp"
    if ext == ".md":
        return "markdown"
    if ext in {".yml", ".yaml"}:
        return "yaml"
    if ext == ".json":
        return "json"
    return ext.lstrip(".")


def _chunk_by_lines(
    *,
    file_path: Path,
    lines: List[str],
    start_line: int,
    end_line: int,
    last_modified: float,
    language: str,
    symbols: Optional[List[str]] = None,
) -> List[CodeChunk]:
    """Chunk a line range using the existing size-based approach."""
    symbols = symbols or []
    chunks: List[CodeChunk] = []

    current_chunk_lines: List[str] = []
    current_size = 0
    chunk_start = start_line

    for idx in range(start_line, end_line + 1):
        line = lines[idx - 1]
        line_size = len(line) + 1

        if current_size + line_size > MAX_CHUNK_SIZE and current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append(
                CodeChunk(
                    file_path=str(file_path),
                    start_line=chunk_start,
                    end_line=idx - 1,
                    content=chunk_content,
                    chunk_hash=hash_content(f"{file_path}:{chunk_start}:{chunk_content}"),
                    symbols=symbols,
                    language=language,
                    last_modified=last_modified,
                )
            )

            overlap_lines = current_chunk_lines[-3:] if len(current_chunk_lines) > 3 else []
            current_chunk_lines = overlap_lines
            current_size = sum(len(l) + 1 for l in overlap_lines)
            chunk_start = max(start_line, idx - len(overlap_lines))

        current_chunk_lines.append(line)
        current_size += line_size

    if current_chunk_lines:
        chunk_content = "\n".join(current_chunk_lines)
        chunks.append(
            CodeChunk(
                file_path=str(file_path),
                start_line=chunk_start,
                end_line=end_line,
                content=chunk_content,
                chunk_hash=hash_content(f"{file_path}:{chunk_start}:{chunk_content}"),
                symbols=symbols,
                language=language,
                last_modified=last_modified,
            )
        )

    return chunks


def _chunk_python_ast(file_path: Path, content: str, *, last_modified: float) -> Optional[List[CodeChunk]]:
    """Chunk Python by top-level defs/classes, with line-based fallback for gaps/large blocks."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    lines = content.split("\n")
    language = "python"

    spans: List[tuple[int, int, List[str]]] = []
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", None)
            if not end:
                continue
            spans.append((int(node.lineno), int(end), [node.name]))
        elif isinstance(node, ast.ClassDef):
            end = getattr(node, "end_lineno", None)
            if not end:
                continue
            method_names = [
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ][:10]
            spans.append((int(node.lineno), int(end), [node.name, *method_names]))

    if not spans:
        return None

    spans.sort(key=lambda x: (x[0], x[1]))
    chunks: List[CodeChunk] = []

    cur = 1
    total_lines = len(lines)
    for start, end, symbols in spans:
        start = max(1, start)
        end = min(total_lines, end)
        if start > cur:
            gap = "\n".join(lines[cur - 1:start - 1])
            if gap.strip():
                chunks.extend(
                    _chunk_by_lines(
                        file_path=file_path,
                        lines=lines,
                        start_line=cur,
                        end_line=start - 1,
                        last_modified=last_modified,
                        language=language,
                        symbols=[],
                    )
                )

        block_text = "\n".join(lines[start - 1:end])
        if not block_text.strip():
            cur = max(cur, end + 1)
            continue

        if len(block_text) > MAX_CHUNK_SIZE:
            chunks.extend(
                _chunk_by_lines(
                    file_path=file_path,
                    lines=lines,
                    start_line=start,
                    end_line=end,
                    last_modified=last_modified,
                    language=language,
                    symbols=symbols,
                )
            )
        else:
            chunks.append(
                CodeChunk(
                    file_path=str(file_path),
                    start_line=start,
                    end_line=end,
                    content=block_text,
                    chunk_hash=hash_content(f"{file_path}:{start}:{block_text}"),
                    symbols=symbols,
                    language=language,
                    last_modified=last_modified,
                )
            )

        cur = max(cur, end + 1)

    if cur <= total_lines:
        tail = "\n".join(lines[cur - 1:])
        if tail.strip():
            chunks.extend(
                _chunk_by_lines(
                    file_path=file_path,
                    lines=lines,
                    start_line=cur,
                    end_line=total_lines,
                    last_modified=last_modified,
                    language=language,
                    symbols=[],
                )
            )

    return chunks


# Files to skip
SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock", "composer.lock",
}


def should_index_file(file_path: Path) -> bool:
    """Check if file should be indexed."""
    if file_path.name in SKIP_FILES:
        return False
    return file_path.suffix.lower() in CODE_EXTENSIONS


def should_skip_dir(dir_name: str) -> bool:
    """Check if directory should be skipped."""
    return dir_name in SKIP_DIRS or dir_name.startswith(".")


def chunk_file(file_path: Path) -> List[CodeChunk]:
    """Split a file into chunks."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    if not content.strip():
        return []

    try:
        last_modified = float(file_path.stat().st_mtime)
    except Exception:
        last_modified = 0.0

    language = _language_for_path(file_path)

    if file_path.suffix.lower() == ".py":
        ast_chunks = _chunk_python_ast(file_path, content, last_modified=last_modified)
        if ast_chunks is not None:
            return ast_chunks

    lines = content.split("\n")
    return _chunk_by_lines(
        file_path=file_path,
        lines=lines,
        start_line=1,
        end_line=len(lines),
        last_modified=last_modified,
        language=language,
        symbols=[],
    )


def walk_codebase(root_path: Path) -> Generator[Path, None, None]:
    """Walk codebase and yield files to index."""
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Filter out skip directories
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]

        for filename in filenames:
            file_path = Path(dirpath) / filename
            if should_index_file(file_path):
                yield file_path


def chunk_codebase(root_path: Path) -> Generator[CodeChunk, None, None]:
    """Chunk entire codebase."""
    for file_path in walk_codebase(root_path):
        for chunk in chunk_file(file_path):
            yield chunk
