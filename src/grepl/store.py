"""ChromaDB vector store."""

import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import chromadb
from chromadb.config import Settings

from .chunker import CodeChunk, build_rich_text
from .embedder import get_embeddings, get_embedding, check_ollama, check_model
from .planner import is_identifier_like_query
from .query_expander import get_expanded_queries, is_natural_language_query
from .code_graph import index_from_chunks as index_code_graph, clear_file as clear_code_graph_file

# Store data in ~/.grepl
GREPPY_DIR = Path.home() / ".grepl"
CHROMA_DIR = GREPPY_DIR / "chroma"
METADATA_DIR = GREPPY_DIR / "metadata"

SEMANTIC_SCORE_FLOOR = 0.18
SEMANTIC_SCORE_CEILING = 0.95
SEMANTIC_EXPANSION_PENALTY = 0.9


def normalize_semantic_scores(scores: List[float]) -> List[float]:
    """Normalize semantic scores to [0,1] per query."""
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score < 1e-6:
        return [max(0.0, min(1.0, s)) for s in scores]
    return [max(0.0, min(1.0, (s - min_score) / (max_score - min_score))) for s in scores]


def apply_semantic_confidence_bounds(
    score: float,
    *,
    floor: float = SEMANTIC_SCORE_FLOOR,
    ceiling: float = SEMANTIC_SCORE_CEILING,
) -> float:
    """Clamp low-confidence semantic scores and cap the ceiling."""
    if score < floor:
        return 0.0
    return min(score, ceiling)


def _distance_to_score(distance: float) -> float:
    """Convert Chroma distance to a bounded similarity score."""
    if distance <= 2.0:
        score = 1.0 - (distance / 2.0)
    else:
        score = 1.0 / (1.0 + distance)
    return max(0.0, min(1.0, score))


def get_collection_name(project_path: Path) -> str:
    """Generate collection name from project path."""
    # Use path hash to create unique collection name
    import hashlib
    path_hash = hashlib.md5(str(project_path.resolve()).encode()).hexdigest()[:8]
    name = project_path.name.replace("-", "_").replace(".", "_")[:20]
    return f"{name}_{path_hash}"


def get_client() -> chromadb.Client:
    """Get ChromaDB client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(project_path: Path):
    """Get or create collection for project."""
    client = get_client()
    name = get_collection_name(project_path)
    return client.get_or_create_collection(
        name=name,
        metadata={
            "project_path": str(project_path.resolve()),
            # Better default semantics for embedding search.
            # See Chroma docs: HNSW space can be 'cosine', 'l2', 'ip'.
            "hnsw:space": "cosine",
        },
    )


def has_index(project_path: Path) -> bool:
    """Check if project has an index."""
    try:
        collection = get_collection(project_path)
        return collection.count() > 0
    except Exception:
        return False


def index_chunks(project_path: Path, chunks: List[CodeChunk], batch_size: int = 50) -> int:
    """Index chunks into ChromaDB."""
    collection = get_collection(project_path)

    # Clear existing data
    try:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

    total_indexed = 0

    # Process in batches
    batch = []
    for chunk in chunks:
        batch.append(chunk)

        if len(batch) >= batch_size:
            _index_batch(collection, batch, project_path)
            total_indexed += len(batch)
            batch = []

    # Index remaining
    if batch:
        _index_batch(collection, batch, project_path)
        total_indexed += len(batch)

    update_code_graph(chunks)

    return total_indexed


def _index_batch(collection, chunks: List[CodeChunk], project_root: Optional[Path] = None):
    """Index a batch of chunks.

    Uses enriched text (with metadata) for embedding while storing raw content for display.
    """
    # Build rich text for embedding (includes language, file path, symbols)
    rich_texts = [build_rich_text(c, project_root) for c in chunks]
    embeddings = get_embeddings(rich_texts)

    # Store raw content as documents (for display), but embed rich text
    raw_texts = [c.content for c in chunks]

    collection.add(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=raw_texts,
        metadatas=[
            {
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                # Chroma metadata must be scalar; store symbols as JSON.
                "symbols": json.dumps(getattr(c, "symbols", []) or []),
                "language": getattr(c, "language", "") or "",
                "last_modified": float(getattr(c, "last_modified", 0.0) or 0.0),
                "parent_symbol": getattr(c, "parent_symbol", "") or "",
                "chunk_type": getattr(c, "chunk_type", "") or "",
                "imports": json.dumps(getattr(c, "imports", []) or []),
                "calls": json.dumps(getattr(c, "calls", []) or []),
                "inherits": json.dumps(getattr(c, "inherits", []) or []),
            }
            for c in chunks
        ],
    )


def search(
    project_path: Path,
    query: str,
    limit: int = 10,
    use_expansion: bool = True,
    llm_expand: bool = False,
) -> List[dict]:
    """Search indexed codebase.

    Args:
        project_path: Path to the project root
        query: Search query
        limit: Maximum number of results
        use_expansion: Whether to expand natural language queries

    Returns:
        List of search results with scores
    """
    collection = get_collection(project_path)

    if collection.count() == 0:
        return []

    # Expand query for natural language searches
    if use_expansion and is_natural_language_query(query) and not is_identifier_like_query(query):
        queries = get_expanded_queries(query, llm_expand=llm_expand)[:3]
        query_embeddings = get_embeddings(queries)

        # Query with each embedding and merge results
        all_results: Dict[str, dict] = {}
        for i, qe in enumerate(query_embeddings):
            results = collection.query(
                query_embeddings=[qe],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
            # Merge results, keeping best score per chunk
            for j in range(len(results["ids"][0])):
                chunk_id = results["ids"][0][j]
                distance = float(results["distances"][0][j])
                raw_score = _distance_to_score(distance)
                penalty = 1.0 if i == 0 else SEMANTIC_EXPANSION_PENALTY
                weighted_score = raw_score * penalty

                if chunk_id not in all_results or weighted_score > all_results[chunk_id]["score_weighted"]:
                    all_results[chunk_id] = {
                        "id": chunk_id,
                        "document": results["documents"][0][j],
                        "metadata": results["metadatas"][0][j],
                        "score_raw": raw_score,
                        "score_weighted": weighted_score,
                    }

        # Normalize semantic scores per query
        weighted_scores = [r["score_weighted"] for r in all_results.values()]
        normalized_scores = normalize_semantic_scores(weighted_scores)
        for r, normalized in zip(all_results.values(), normalized_scores):
            r["score_normalized"] = normalized
            r["score"] = apply_semantic_confidence_bounds(normalized)

        # Sort by score and take top results
        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)[:limit]

        # Format results
        formatted = []
        for r in sorted_results:
            meta = r["metadata"] or {}
            symbols = meta.get("symbols")
            if isinstance(symbols, str):
                try:
                    symbols = json.loads(symbols)
                except Exception:
                    symbols = [s for s in symbols.split(",") if s]
            if not isinstance(symbols, list):
                symbols = []

            formatted.append({
                "file_path": meta.get("file_path"),
                "start_line": meta.get("start_line"),
                "end_line": meta.get("end_line"),
                "content": r["document"],
                "score": r["score"],
                "score_raw": r.get("score_raw", 0.0),
                "score_normalized": r.get("score_normalized", r["score"]),
                "symbols": symbols,
                "language": meta.get("language") or "",
                "last_modified": float(meta.get("last_modified") or 0.0),
            })

        return formatted

    # Single query (no expansion)
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    formatted = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i] or {}

        symbols = meta.get("symbols")
        if isinstance(symbols, str):
            try:
                symbols = json.loads(symbols)
            except Exception:
                symbols = [s for s in symbols.split(",") if s]
        if not isinstance(symbols, list):
            symbols = []

        distance = float(results["distances"][0][i])
        raw_score = _distance_to_score(distance)

        formatted.append({
            "file_path": meta.get("file_path"),
            "start_line": meta.get("start_line"),
            "end_line": meta.get("end_line"),
            "content": results["documents"][0][i],
            "score_raw": raw_score,
            "symbols": symbols,
            "language": meta.get("language") or "",
            "last_modified": float(meta.get("last_modified") or 0.0),
        })

    scores = [r["score_raw"] for r in formatted]
    normalized_scores = normalize_semantic_scores(scores)
    for r, normalized in zip(formatted, normalized_scores):
        r["score_normalized"] = normalized
        r["score"] = apply_semantic_confidence_bounds(normalized)

    return formatted


def clear_index(project_path: Path):
    """Clear index for project."""
    client = get_client()
    name = get_collection_name(project_path)
    try:
        try:
            collection = get_collection(project_path)
            data = collection.get(include=["metadatas"])
            metadatas = data.get("metadatas", []) if data else []
            for meta in metadatas:
                if not meta:
                    continue
                fp = meta.get("file_path")
                if fp:
                    clear_code_graph_file(fp)
        except Exception:
            pass
        client.delete_collection(name)
    except Exception:
        pass


def delete_chunks_for_file(project_path: Path, file_path: str):
    """Delete all chunks for a specific file from ChromaDB."""
    collection = get_collection(project_path)
    collection.delete(where={"file_path": file_path})


def delete_code_graph_for_file(file_path: str) -> None:
    """Delete graph data for a specific file."""
    clear_code_graph_file(file_path)


def upsert_file_chunks(project_path: Path, chunks: List[CodeChunk]) -> int:
    """Add chunks without clearing existing data."""
    if not chunks:
        return 0
    collection = get_collection(project_path)
    _index_batch(collection, chunks, project_path)
    update_code_graph(chunks)
    return len(chunks)


def update_code_graph(chunks: List[CodeChunk]) -> None:
    """Update the code graph for a set of chunks."""
    if not chunks:
        return
    index_code_graph(chunks)


def get_stats(project_path: Path) -> dict:
    """Get index stats."""
    try:
        collection = get_collection(project_path)
        return {
            "exists": True,
            "chunks": collection.count(),
            "project": str(project_path.resolve()),
        }
    except Exception:
        return {
            "exists": False,
            "chunks": 0,
            "project": str(project_path.resolve()),
        }


def get_detailed_stats(project_path: Path) -> dict:
    """Get comprehensive index statistics for agents.

    Returns machine-readable stats including:
    - indexed: Whether the index has content
    - chunks: Number of indexed chunks
    - files: Number of unique files indexed
    - lastIndexedAt: ISO timestamp of last index (or None)
    - workspaceFingerprint: Collection name for this workspace
    - ollamaRunning: Whether Ollama is accessible
    - modelAvailable: Whether nomic-embed-text model is available
    - semanticReady: Whether semantic search will work
    - semanticReadyReason: Why semantic isn't ready (if applicable)
    """
    resolved_path = project_path.resolve()
    fingerprint = get_collection_name(project_path)

    # Check Ollama status
    ollama_running = check_ollama()
    model_available = check_model() if ollama_running else False

    # Default values
    result = {
        "indexed": False,
        "chunks": 0,
        "files": 0,
        "lastIndexedAt": None,
        "workspaceFingerprint": fingerprint,
        "projectPath": str(resolved_path),
        "ollamaRunning": ollama_running,
        "modelAvailable": model_available,
        "semanticReady": False,
        "semanticReadyReason": None,
    }

    try:
        collection = get_collection(project_path)
        chunk_count = collection.count()

        if chunk_count > 0:
            result["indexed"] = True
            result["chunks"] = chunk_count

            # Get unique file count and last modified from metadata
            try:
                all_data = collection.get(include=["metadatas"])
                metadatas = all_data.get("metadatas", [])
                if metadatas:
                    unique_files = set()
                    max_modified = 0.0
                    for meta in metadatas:
                        if meta:
                            fp = meta.get("file_path")
                            if fp:
                                unique_files.add(fp)
                            lm = meta.get("last_modified", 0.0)
                            if lm and float(lm) > max_modified:
                                max_modified = float(lm)

                    result["files"] = len(unique_files)
                    if max_modified > 0:
                        result["lastIndexedAt"] = datetime.fromtimestamp(max_modified).isoformat()
            except Exception:
                pass

    except Exception:
        pass

    # Determine semantic readiness
    if not ollama_running:
        result["semanticReadyReason"] = "ollama_not_running"
    elif not model_available:
        result["semanticReadyReason"] = "model_not_available"
    elif not result["indexed"]:
        result["semanticReadyReason"] = "not_indexed"
    else:
        result["semanticReady"] = True

    return result


def check_semantic_ready(project_path: Path) -> Tuple[bool, Optional[str]]:
    """Check if semantic search is ready.

    Returns:
        (ready, reason_if_not)
        - (True, None) - ready
        - (False, "ollama_not_running")
        - (False, "model_not_available")
        - (False, "not_indexed")
    """
    if not check_ollama():
        return False, "ollama_not_running"
    if not check_model():
        return False, "model_not_available"
    if not has_index(project_path):
        return False, "not_indexed"
    return True, None


def _get_metadata_path(project_path: Path) -> Path:
    """Get metadata file path for a project."""
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    collection_name = get_collection_name(project_path)
    return METADATA_DIR / f"{collection_name}.json"


def _compute_file_hash(file_path: Path) -> str:
    """Compute content hash for a file."""
    try:
        content = file_path.read_bytes()
        return hashlib.md5(content).hexdigest()
    except Exception:
        return ""


def store_index_metadata(project_path: Path, files_metadata: Dict[str, dict]) -> None:
    """Store metadata for incremental updates.

    Args:
        project_path: Root path of the project
        files_metadata: Dict mapping file_path to {mtime, size, content_hash}
    """
    metadata_path = _get_metadata_path(project_path)
    metadata = {
        "project_path": str(project_path.resolve()),
        "last_indexed_at": datetime.now().isoformat(),
        "files": files_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))


def load_index_metadata(project_path: Path) -> Optional[dict]:
    """Load stored metadata for a project."""
    metadata_path = _get_metadata_path(project_path)
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text())
    except Exception:
        return None


def get_changed_files(
    project_path: Path, current_files: Dict[str, dict]
) -> Tuple[List[str], List[str], List[str]]:
    """Compare current files against stored metadata.

    Args:
        project_path: Root path of the project
        current_files: Dict mapping file_path to {mtime, size, content_hash}

    Returns:
        (new_files, modified_files, deleted_files)
    """
    metadata = load_index_metadata(project_path)
    if not metadata:
        # No metadata = all files are new
        return list(current_files.keys()), [], []

    stored_files = metadata.get("files", {})

    new_files = []
    modified_files = []
    deleted_files = []

    # Check for new and modified files
    for file_path, file_meta in current_files.items():
        if file_path not in stored_files:
            new_files.append(file_path)
        else:
            stored_meta = stored_files[file_path]
            # Check if file changed (compare mtime, size, or hash)
            if (
                file_meta.get("mtime") != stored_meta.get("mtime")
                or file_meta.get("size") != stored_meta.get("size")
                or file_meta.get("content_hash") != stored_meta.get("content_hash")
            ):
                modified_files.append(file_path)

    # Check for deleted files
    for file_path in stored_files:
        if file_path not in current_files:
            deleted_files.append(file_path)

    return new_files, modified_files, deleted_files
