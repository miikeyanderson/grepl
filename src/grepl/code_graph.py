from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


GRAPH_PATH = Path.home() / ".grepl" / "code_graph.db"


@dataclass(frozen=True)
class CodeNode:
    id: str
    symbol: str
    file_path: str
    node_type: str


@dataclass(frozen=True)
class CodeEdge:
    source: str
    target: str
    edge_type: str
    source_file: str


def _connect() -> sqlite3.Connection:
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(GRAPH_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            file_path TEXT NOT NULL,
            node_type TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            edge_type TEXT NOT NULL,
            source_file TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source_file ON edges(source_file)")
    return conn


def _node_id(symbol: str, file_path: str, node_type: str) -> str:
    raw = f"{file_path}|{symbol}|{node_type}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def clear_file(file_path: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM edges WHERE source_file = ?", (file_path,))
        conn.execute("DELETE FROM nodes WHERE file_path = ?", (file_path,))
        conn.commit()


def upsert_nodes_edges(nodes: Iterable[CodeNode], edges: Iterable[CodeEdge]) -> None:
    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO nodes (id, symbol, file_path, node_type) VALUES (?, ?, ?, ?)",
            [(n.id, n.symbol, n.file_path, n.node_type) for n in nodes],
        )
        conn.executemany(
            "INSERT INTO edges (source, target, edge_type, source_file) VALUES (?, ?, ?, ?)",
            [(e.source, e.target, e.edge_type, e.source_file) for e in edges],
        )
        conn.commit()


def index_from_chunks(chunks: Iterable) -> None:
    chunks_by_file: Dict[str, List] = {}
    for chunk in chunks:
        chunks_by_file.setdefault(chunk.file_path, []).append(chunk)

    for file_path, file_chunks in chunks_by_file.items():
        clear_file(file_path)

        nodes: List[CodeNode] = []
        edges: List[CodeEdge] = []
        file_node = CodeNode(
            id=_node_id(file_path, file_path, "file"),
            symbol=file_path,
            file_path=file_path,
            node_type="file",
        )
        nodes.append(file_node)

        for chunk in file_chunks:
            symbols = getattr(chunk, "symbols", []) or []
            chunk_type = getattr(chunk, "chunk_type", "") or "symbol"
            imports = getattr(chunk, "imports", []) or []
            calls = getattr(chunk, "calls", []) or []
            inherits = getattr(chunk, "inherits", []) or []

            for symbol in symbols:
                nodes.append(CodeNode(
                    id=_node_id(symbol, file_path, chunk_type),
                    symbol=symbol,
                    file_path=file_path,
                    node_type=chunk_type,
                ))

            for imported in imports:
                edges.append(CodeEdge(
                    source=file_node.symbol,
                    target=imported,
                    edge_type="import",
                    source_file=file_path,
                ))

            source_symbol = symbols[0] if symbols else file_node.symbol
            for call in calls:
                edges.append(CodeEdge(
                    source=source_symbol,
                    target=call,
                    edge_type="call",
                    source_file=file_path,
                ))

            if symbols:
                for base in inherits:
                    edges.append(CodeEdge(
                        source=symbols[0],
                        target=base,
                        edge_type="inherits",
                        source_file=file_path,
                    ))

        upsert_nodes_edges(nodes, edges)


def find_callers(symbol: str) -> List[Tuple[str, str]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT source, source_file FROM edges WHERE target = ? AND edge_type = 'call'",
            (symbol,),
        ).fetchall()
    return [(row[0], row[1]) for row in rows]


def find_implementations(symbol: str) -> List[Tuple[str, str]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT source, source_file FROM edges WHERE target = ? AND edge_type = 'inherits'",
            (symbol,),
        ).fetchall()
    return [(row[0], row[1]) for row in rows]


def imports_for_file(file_path: str) -> List[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT target FROM edges WHERE source_file = ? AND edge_type = 'import'",
            (file_path,),
        ).fetchall()
    return [row[0] for row in rows]
