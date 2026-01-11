from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from tree_sitter_languages import get_parser
except Exception:  # pragma: no cover - optional dependency
    get_parser = None


MAX_CHUNK_SIZE = 2000


@dataclass
class SemanticChunk:
    file_path: str
    start_line: int
    end_line: int
    content: str
    chunk_hash: str
    symbols: List[str] = field(default_factory=list)
    language: str = ""
    last_modified: float = 0.0
    parent_symbol: str = ""
    chunk_type: str = ""
    docstring: str = ""
    imports: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    inherits: List[str] = field(default_factory=list)


_LANGUAGE_PARSER = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
}


CLASS_NODE_TYPES = {
    "python": {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration", "interface_declaration"},
    "tsx": {"class_declaration", "interface_declaration"},
    "java": {"class_declaration", "interface_declaration"},
    "kotlin": {"class_declaration", "object_declaration", "interface_declaration"},
    "swift": {"class_declaration", "struct_declaration", "enum_declaration", "protocol_declaration", "extension_declaration"},
    "go": {"type_declaration"},
    "rust": {"struct_item", "enum_item", "trait_item", "impl_item"},
    "c": set(),
    "cpp": {"class_specifier", "struct_specifier"},
    "ruby": {"class", "module"},
}

FUNCTION_NODE_TYPES = {
    "python": {"function_definition"},
    "javascript": {"function_declaration", "method_definition", "arrow_function"},
    "typescript": {"function_declaration", "method_definition", "arrow_function"},
    "tsx": {"function_declaration", "method_definition", "arrow_function"},
    "java": {"method_declaration", "constructor_declaration"},
    "kotlin": {"function_declaration"},
    "swift": {"function_declaration", "initializer_declaration"},
    "go": {"function_declaration", "method_declaration"},
    "rust": {"function_item"},
    "c": {"function_definition"},
    "cpp": {"function_definition"},
    "ruby": {"method"},
}

CALL_NODE_TYPES = {
    "python": {"call"},
    "javascript": {"call_expression"},
    "typescript": {"call_expression"},
    "tsx": {"call_expression"},
    "java": {"method_invocation"},
    "kotlin": {"call_expression"},
    "swift": {"call_expression"},
    "go": {"call_expression"},
    "rust": {"call_expression"},
    "c": {"call_expression"},
    "cpp": {"call_expression"},
    "ruby": {"call"},
}


def _parser_for_path(file_path: Path) -> Optional[str]:
    return _LANGUAGE_PARSER.get(file_path.suffix.lower())


def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="ignore")


def _node_name(node, source_bytes: bytes) -> str:
    name_node = node.child_by_field_name("name") if hasattr(node, "child_by_field_name") else None
    if name_node is not None:
        return _node_text(name_node, source_bytes).strip()
    for child in node.children:
        if child.type in {"identifier", "type_identifier", "property_identifier"}:
            return _node_text(child, source_bytes).strip()
    return ""


def _extract_imports(language: str, content: str) -> List[str]:
    imports: List[str] = []
    if language == "python":
        for m in re.finditer(r"^\s*import\s+([A-Za-z0-9_\.]+)", content, re.MULTILINE):
            imports.append(m.group(1))
        for m in re.finditer(r"^\s*from\s+([A-Za-z0-9_\.]+)\s+import", content, re.MULTILINE):
            imports.append(m.group(1))
    elif language in {"javascript", "typescript", "tsx"}:
        for m in re.finditer(r"import\s+.*?from\s+[\"']([^\"']+)[\"']", content):
            imports.append(m.group(1))
        for m in re.finditer(r"require\([\"']([^\"']+)[\"']\)", content):
            imports.append(m.group(1))
    elif language == "go":
        for m in re.finditer(r"import\s+\"([^\"]+)\"", content):
            imports.append(m.group(1))
    elif language in {"java", "kotlin"}:
        for m in re.finditer(r"^\s*import\s+([A-Za-z0-9_\.]+)", content, re.MULTILINE):
            imports.append(m.group(1))
    elif language == "rust":
        for m in re.finditer(r"^\s*use\s+([A-Za-z0-9_:]+)", content, re.MULTILINE):
            imports.append(m.group(1))
    elif language == "swift":
        for m in re.finditer(r"^\s*import\s+([A-Za-z0-9_]+)", content, re.MULTILINE):
            imports.append(m.group(1))
    elif language in {"c", "cpp"}:
        for m in re.finditer(r"#include\s+[<\"]([^>\"]+)[>\"]", content):
            imports.append(m.group(1))
    return sorted(set(imports))


def _extract_docstring(lines: List[str], start_line: int, language: str) -> str:
    idx = start_line
    if idx >= len(lines):
        return ""

    if language == "python":
        if idx < len(lines):
            line = lines[idx].lstrip()
            if line.startswith("\"\"") or line.startswith("\'\'"):
                return line.strip()
            if line.startswith("\"") or line.startswith("\'"):
                return line.strip()
        return ""

    # Fallback: collect contiguous comment lines above start_line
    comments = []
    i = start_line - 1
    while i >= 0:
        stripped = lines[i].strip()
        if not stripped:
            break
        if stripped.startswith(("//", "#", "/*", "*")):
            comments.append(stripped)
            i -= 1
            continue
        break
    return " ".join(reversed(comments))


def _extract_calls(language: str, node, source_bytes: bytes) -> List[str]:
    call_types = CALL_NODE_TYPES.get(language, set())
    calls: List[str] = []

    def walk(n):
        if n.type in call_types:
            target = n.child_by_field_name("function") if hasattr(n, "child_by_field_name") else None
            if target is None:
                for child in n.children:
                    if child.type in {"identifier", "property_identifier", "field_identifier", "member_expression"}:
                        target = child
                        break
            if target is not None:
                name = _node_text(target, source_bytes).strip()
                if name:
                    calls.append(name.split("(")[0])
        for child in n.children:
            walk(child)

    walk(node)
    return sorted(set(calls))


def _extract_inherits(language: str, node, source_bytes: bytes) -> List[str]:
    inherits: List[str] = []
    if language == "python":
        for child in node.children:
            if child.type == "argument_list":
                text = _node_text(child, source_bytes)
                inherits.extend([c.strip() for c in text.strip("() ").split(",") if c.strip()])
    elif language in {"javascript", "typescript", "tsx"}:
        for child in node.children:
            if child.type == "extends_clause":
                text = _node_text(child, source_bytes)
                inherits.append(text.replace("extends", "").strip())
    elif language in {"java", "kotlin"}:
        for child in node.children:
            if child.type in {"superclass", "super_interfaces"}:
                inherits.append(_node_text(child, source_bytes).strip())
    elif language == "swift":
        for child in node.children:
            if child.type == "inheritance_clause":
                text = _node_text(child, source_bytes)
                inherits.extend([c.strip() for c in text.replace(":", "").split(",") if c.strip()])
    return sorted(set([i for i in inherits if i]))


def chunk_file_with_treesitter(
    file_path: Path,
    content: str,
    *,
    last_modified: float,
) -> Optional[List[SemanticChunk]]:
    if get_parser is None:
        return None

    parser_name = _parser_for_path(file_path)
    if not parser_name:
        return None

    try:
        parser = get_parser(parser_name)
    except Exception:
        return None

    source_bytes = content.encode("utf-8", errors="ignore")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    language = parser_name
    lines = content.split("\n")
    imports = _extract_imports(language, content)

    class_types = CLASS_NODE_TYPES.get(language, set())
    func_types = FUNCTION_NODE_TYPES.get(language, set())

    chunks: List[SemanticChunk] = []

    def walk(node, class_stack: List[str]):
        node_type = node.type
        is_class = node_type in class_types
        is_func = node_type in func_types

        if is_class:
            class_name = _node_name(node, source_bytes)
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            docstring = _extract_docstring(lines, start_line, language)
            methods = []
            for child in node.children:
                if child.type in func_types:
                    method_name = _node_name(child, source_bytes)
                    if method_name:
                        methods.append(method_name)
            content_text = "\n".join(lines[start_line - 1:end_line])
            if content_text.strip():
                chunks.append(SemanticChunk(
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    content=content_text,
                    chunk_hash="",
                    symbols=[s for s in [class_name, *methods] if s],
                    language=language,
                    last_modified=last_modified,
                    parent_symbol="",
                    chunk_type="class",
                    docstring=docstring,
                    imports=imports,
                    calls=_extract_calls(language, node, source_bytes),
                    inherits=_extract_inherits(language, node, source_bytes),
                ))
            class_stack.append(class_name)
            for child in node.children:
                walk(child, class_stack)
            class_stack.pop()
            return

        if is_func:
            func_name = _node_name(node, source_bytes)
            parent = class_stack[-1] if class_stack else ""
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            docstring = _extract_docstring(lines, start_line, language)
            content_text = "\n".join(lines[start_line - 1:end_line])
            if content_text.strip():
                chunks.append(SemanticChunk(
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    content=content_text,
                    chunk_hash="",
                    symbols=[s for s in [func_name] if s],
                    language=language,
                    last_modified=last_modified,
                    parent_symbol=parent,
                    chunk_type="method" if parent else "function",
                    docstring=docstring,
                    imports=imports,
                    calls=_extract_calls(language, node, source_bytes),
                    inherits=[],
                ))
            return

        for child in node.children:
            walk(child, class_stack)

    walk(root, [])

    if not chunks:
        return None

    # Add gap chunks for top-level code not covered by functions/classes
    spans = sorted((c.start_line, c.end_line) for c in chunks)
    cur = 1
    total_lines = len(lines)
    for start, end in spans:
        if start > cur:
            gap_text = "\n".join(lines[cur - 1:start - 1])
            if gap_text.strip():
                chunks.append(SemanticChunk(
                    file_path=str(file_path),
                    start_line=cur,
                    end_line=start - 1,
                    content=gap_text,
                    chunk_hash="",
                    symbols=[],
                    language=language,
                    last_modified=last_modified,
                    parent_symbol="",
                    chunk_type="block",
                    docstring="",
                    imports=imports,
                    calls=[],
                    inherits=[],
                ))
        cur = max(cur, end + 1)
    if cur <= total_lines:
        tail_text = "\n".join(lines[cur - 1:])
        if tail_text.strip():
            chunks.append(SemanticChunk(
                file_path=str(file_path),
                start_line=cur,
                end_line=total_lines,
                content=tail_text,
                chunk_hash="",
                symbols=[],
                language=language,
                last_modified=last_modified,
                parent_symbol="",
                chunk_type="block",
                docstring="",
                imports=imports,
                calls=[],
                inherits=[],
            ))

    # Fill in chunk hashes and split large chunks
    finalized: List[SemanticChunk] = []
    for chunk in chunks:
        if len(chunk.content) > MAX_CHUNK_SIZE:
            lines_chunk = chunk.content.split("\n")
            start = chunk.start_line
            for i in range(0, len(lines_chunk), 200):
                sub_lines = lines_chunk[i:i + 200]
                if not sub_lines:
                    continue
                sub_content = "\n".join(sub_lines)
                sub_start = start + i
                sub_end = sub_start + len(sub_lines) - 1
                finalized.append(SemanticChunk(
                    file_path=chunk.file_path,
                    start_line=sub_start,
                    end_line=sub_end,
                    content=sub_content,
                    chunk_hash="",
                    symbols=chunk.symbols,
                    language=chunk.language,
                    last_modified=chunk.last_modified,
                    parent_symbol=chunk.parent_symbol,
                    chunk_type=chunk.chunk_type,
                    docstring=chunk.docstring,
                    imports=chunk.imports,
                    calls=chunk.calls,
                    inherits=chunk.inherits,
                ))
        else:
            finalized.append(chunk)

    return finalized
