"""Code chunking - splits files into searchable chunks."""

import ast
import os
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Generator, Optional, Tuple

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


def build_rich_text(chunk: "CodeChunk", project_root: Optional[Path] = None) -> str:
    """Build enriched text for embedding that includes semantic context.

    Instead of embedding raw code, we prepend metadata that helps the
    embedding model understand the semantic meaning of the code.
    """
    parts = []

    # Language tag
    if chunk.language:
        parts.append(f"[LANG={chunk.language}]")

    # File path (use relative path if project_root provided)
    file_path = chunk.file_path
    if project_root:
        try:
            file_path = str(Path(chunk.file_path).relative_to(project_root))
        except ValueError:
            pass
    parts.append(f"[FILE={file_path}]")

    # Line range
    parts.append(f"[LINES={chunk.start_line}-{chunk.end_line}]")

    # Symbols (function/class names)
    if chunk.symbols:
        symbols_str = ", ".join(chunk.symbols[:5])  # Limit to 5 symbols
        parts.append(f"[SYMBOLS={symbols_str}]")

    # Add header
    header = " ".join(parts)

    # Combine header with content
    return f"{header}\n{chunk.content}"


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


# Language-specific patterns for extracting function/class definitions
SYMBOL_PATTERNS = {
    "swift": [
        (r'^\s*(?:public\s+|private\s+|internal\s+|fileprivate\s+|open\s+)?(?:final\s+)?class\s+(\w+)', 'class'),
        (r'^\s*(?:public\s+|private\s+|internal\s+|fileprivate\s+|open\s+)?struct\s+(\w+)', 'struct'),
        (r'^\s*(?:public\s+|private\s+|internal\s+|fileprivate\s+|open\s+)?enum\s+(\w+)', 'enum'),
        (r'^\s*(?:public\s+|private\s+|internal\s+|fileprivate\s+|open\s+)?(?:@\w+\s+)*func\s+(\w+)', 'func'),
        (r'^\s*(?:public\s+|private\s+|internal\s+|fileprivate\s+|open\s+)?protocol\s+(\w+)', 'protocol'),
        (r'^\s*extension\s+(\w+)', 'extension'),
    ],
    "typescript": [
        (r'^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)', 'function'),
        (r'^\s*(?:export\s+)?class\s+(\w+)', 'class'),
        (r'^\s*(?:export\s+)?interface\s+(\w+)', 'interface'),
        (r'^\s*(?:export\s+)?type\s+(\w+)', 'type'),
        (r'^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(', 'arrow_func'),
        (r'^\s*(?:public|private|protected)?\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*[:{]', 'method'),
    ],
    "javascript": [
        (r'^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)', 'function'),
        (r'^\s*(?:export\s+)?class\s+(\w+)', 'class'),
        (r'^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(', 'arrow_func'),
        (r'^\s*(\w+)\s*\([^)]*\)\s*{', 'method'),
    ],
    "go": [
        (r'^\s*func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(', 'func'),
        (r'^\s*type\s+(\w+)\s+struct\s*{', 'struct'),
        (r'^\s*type\s+(\w+)\s+interface\s*{', 'interface'),
        (r'^\s*type\s+(\w+)\s+', 'type'),
    ],
    "rust": [
        (r'^\s*(?:pub\s+)?fn\s+(\w+)', 'fn'),
        (r'^\s*(?:pub\s+)?struct\s+(\w+)', 'struct'),
        (r'^\s*(?:pub\s+)?enum\s+(\w+)', 'enum'),
        (r'^\s*(?:pub\s+)?trait\s+(\w+)', 'trait'),
        (r'^\s*impl(?:<[^>]+>)?\s+(?:(\w+)\s+for\s+)?(\w+)', 'impl'),
        (r'^\s*(?:pub\s+)?mod\s+(\w+)', 'mod'),
    ],
    "java": [
        (r'^\s*(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?class\s+(\w+)', 'class'),
        (r'^\s*(?:public\s+|private\s+|protected\s+)?interface\s+(\w+)', 'interface'),
        (r'^\s*(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?(?:\w+(?:<[^>]+>)?\s+)?(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+(?:,\s*\w+)*)?\s*{', 'method'),
    ],
    "kotlin": [
        (r'^\s*(?:public\s+|private\s+|internal\s+|protected\s+)?(?:data\s+)?class\s+(\w+)', 'class'),
        (r'^\s*(?:public\s+|private\s+|internal\s+|protected\s+)?interface\s+(\w+)', 'interface'),
        (r'^\s*(?:public\s+|private\s+|internal\s+|protected\s+)?(?:suspend\s+)?fun\s+(\w+)', 'fun'),
        (r'^\s*object\s+(\w+)', 'object'),
    ],
}


def _extract_symbols_regex(content: str, language: str) -> List[str]:
    """Extract symbol names from code using regex patterns."""
    patterns = SYMBOL_PATTERNS.get(language, [])
    if not patterns:
        return []

    symbols = []
    for line in content.split('\n'):
        for pattern, _ in patterns:
            match = re.match(pattern, line)
            if match:
                # Get all captured groups (some patterns have multiple)
                for group in match.groups():
                    if group:
                        symbols.append(group)
                break
    return symbols[:10]  # Limit to 10 symbols


def _find_block_boundaries(lines: List[str], language: str) -> List[Tuple[int, int, List[str]]]:
    """Find function/class block boundaries using regex and brace/indent counting.

    Returns list of (start_line, end_line, symbols) tuples (1-indexed).
    """
    patterns = SYMBOL_PATTERNS.get(language, [])
    if not patterns:
        return []

    spans = []
    i = 0
    while i < len(lines):
        line = lines[i]
        matched_symbol = None

        for pattern, _ in patterns:
            match = re.match(pattern, line)
            if match:
                for group in match.groups():
                    if group:
                        matched_symbol = group
                        break
                if matched_symbol:
                    break

        if matched_symbol:
            start_line = i + 1  # 1-indexed
            # Find end of block by counting braces
            brace_count = 0
            found_open = False
            end_line = start_line

            for j in range(i, len(lines)):
                for char in lines[j]:
                    if char == '{':
                        brace_count += 1
                        found_open = True
                    elif char == '}':
                        brace_count -= 1

                if found_open and brace_count == 0:
                    end_line = j + 1  # 1-indexed
                    break
                end_line = j + 1

            # For languages without braces (like Python), use indentation
            if not found_open and language in ("python",):
                base_indent = len(line) - len(line.lstrip())
                for j in range(i + 1, len(lines)):
                    current_line = lines[j]
                    if current_line.strip():
                        current_indent = len(current_line) - len(current_line.lstrip())
                        if current_indent <= base_indent:
                            end_line = j  # Don't include this line
                            break
                    end_line = j + 1

            spans.append((start_line, end_line, [matched_symbol]))
            i = end_line
        else:
            i += 1

    return spans


def _chunk_with_spans(
    *,
    file_path: Path,
    lines: List[str],
    spans: List[Tuple[int, int, List[str]]],
    last_modified: float,
    language: str,
) -> List[CodeChunk]:
    """Chunk file using pre-computed spans (from regex or other parsing).

    Similar to _chunk_python_ast but works with any span list.
    """
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    chunks: List[CodeChunk] = []
    total_lines = len(lines)
    cur = 1

    for start, end, symbols in spans:
        start = max(1, start)
        end = min(total_lines, end)

        # Handle gap before this span
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

        # Large blocks get sub-chunked
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

    # Handle trailing content
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


def load_greplignore(project_path: Path) -> List[str]:
    """Load patterns from .greplignore file."""
    ignore_file = project_path / ".greplignore"
    if not ignore_file.exists():
        return []

    patterns = []
    try:
        with open(ignore_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
    except Exception:
        return []

    return patterns


def _normalize_match_path(path: str) -> str:
    path = path.replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    return path.lstrip("/")


def matches_pattern(path: str, pattern: str, *, is_dir: bool = False) -> bool:
    """Check if path matches gitignore-style pattern."""
    from fnmatch import fnmatch

    raw_pattern = pattern.strip()
    if not raw_pattern:
        return False

    anchored = raw_pattern.startswith("/")
    dir_only = raw_pattern.endswith("/")
    raw_pattern = raw_pattern.strip("/")
    raw_pattern = raw_pattern.replace("\\", "/")
    has_slash = "/" in raw_pattern

    path_norm = _normalize_match_path(path)

    if dir_only:
        parts = [p for p in path_norm.split("/") if p]
        if not is_dir and parts:
            parts = parts[:-1]

        candidates = ["/".join(parts[: i + 1]) for i in range(len(parts))]

        if anchored:
            return any(fnmatch(candidate, raw_pattern) for candidate in candidates)

        if has_slash:
            return any(
                fnmatch(candidate, raw_pattern) or fnmatch(candidate, f"*/{raw_pattern}")
                for candidate in candidates
            )

        return any(fnmatch(Path(candidate).name, raw_pattern) for candidate in candidates)

    if anchored:
        return fnmatch(path_norm, raw_pattern)

    if has_slash:
        return fnmatch(path_norm, raw_pattern) or fnmatch(path_norm, f"*/{raw_pattern}")

    name = Path(path_norm).name
    return (
        fnmatch(name, raw_pattern)
        or fnmatch(path_norm, raw_pattern)
        or fnmatch(path_norm, f"*/{raw_pattern}")
    )


def _relative_match_path(path: Path, project_root: Optional[Path]) -> str:
    if project_root:
        try:
            return path.relative_to(project_root).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def should_index_file(
    file_path: Path,
    greplignore_patterns: Optional[List[str]] = None,
    project_root: Optional[Path] = None,
) -> bool:
    """Check if file should be indexed."""
    # Check extension whitelist
    if file_path.suffix.lower() not in CODE_EXTENSIONS:
        return False

    # Check hardcoded skip files
    if file_path.name in SKIP_FILES:
        return False

    # Check .greplignore patterns
    if greplignore_patterns:
        rel_path = _relative_match_path(file_path, project_root)
        for pattern in greplignore_patterns:
            if matches_pattern(rel_path, pattern, is_dir=False):
                return False

    return True


def should_skip_dir(
    dir_path: Path,
    greplignore_patterns: Optional[List[str]] = None,
    project_root: Optional[Path] = None,
) -> bool:
    """Check if directory should be skipped."""
    # Always skip hardcoded patterns
    if dir_path.name in SKIP_DIRS or dir_path.name.startswith('.'):
        return True

    # Check .greplignore patterns
    if greplignore_patterns:
        for pattern in greplignore_patterns:
            rel_path = _relative_match_path(dir_path, project_root)
            if matches_pattern(rel_path, pattern, is_dir=True):
                return True

    return False


def chunk_file(file_path: Path) -> List[CodeChunk]:
    """Split a file into chunks.

    Uses language-aware chunking:
    - Python: AST-based chunking (most accurate)
    - Swift, TypeScript, JavaScript, Go, Rust, Java, Kotlin: Regex-based
    - Other languages: Line-based with symbol extraction
    """
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
    lines = content.split("\n")

    # Python: Use AST-based chunking (most accurate)
    if file_path.suffix.lower() == ".py":
        ast_chunks = _chunk_python_ast(file_path, content, last_modified=last_modified)
        if ast_chunks is not None:
            return ast_chunks

    # Languages with regex patterns: Use structure-aware chunking
    if language in SYMBOL_PATTERNS:
        spans = _find_block_boundaries(lines, language)
        if spans:
            return _chunk_with_spans(
                file_path=file_path,
                lines=lines,
                spans=spans,
                last_modified=last_modified,
                language=language,
            )
        # Fall back to line-based but extract symbols
        symbols = _extract_symbols_regex(content, language)
        return _chunk_by_lines(
            file_path=file_path,
            lines=lines,
            start_line=1,
            end_line=len(lines),
            last_modified=last_modified,
            language=language,
            symbols=symbols,
        )

    # Other languages: Line-based chunking
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
    greplignore_patterns = load_greplignore(root_path)

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Filter out skip directories
        dirnames[:] = [
            d
            for d in dirnames
            if not should_skip_dir(Path(dirpath) / d, greplignore_patterns, root_path)
        ]

        for filename in filenames:
            file_path = Path(dirpath) / filename
            if should_index_file(file_path, greplignore_patterns, root_path):
                yield file_path


def chunk_codebase(root_path: Path) -> Generator[CodeChunk, None, None]:
    """Chunk entire codebase."""
    for file_path in walk_codebase(root_path):
        for chunk in chunk_file(file_path):
            yield chunk


def collect_file_metadata(root_path: Path) -> dict:
    """Collect metadata (mtime, size, hash) for all indexable files.

    Returns:
        Dict mapping absolute file path to {mtime, size, content_hash}
    """
    import hashlib

    files_metadata = {}
    for file_path in walk_codebase(root_path):
        try:
            stat = file_path.stat()
            # Compute lightweight hash (first 8KB + file size)
            # This is faster than full file hash for large files
            content_hash = ""
            try:
                with open(file_path, "rb") as f:
                    sample = f.read(8192)
                    content_hash = hashlib.md5(sample + str(stat.st_size).encode()).hexdigest()
            except Exception:
                content_hash = ""

            files_metadata[str(file_path.resolve())] = {
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "content_hash": content_hash,
            }
        except Exception:
            continue

    return files_metadata
