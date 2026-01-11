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
from .query_expander import expand_query, is_natural_language_query

# Store data in ~/.grepl
GREPPY_DIR = Path.home() / ".grepl"
CHROMA_DIR = GREPPY_DIR / "chroma"
METADATA_DIR = GREPPY_DIR / "metadata"


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
            }
            for c in chunks
        ],
    )


def search(project_path: Path, query: str, limit: int = 10, use_expansion: bool = True) -> List[dict]:
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
    if use_expansion and is_natural_language_query(query):
        queries = expand_query(query, max_expansions=3)
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
                score = 1.0 - (distance / 2.0) if distance <= 2.0 else 1.0 / (1.0 + distance)
                score = max(0.0, min(1.0, score))

                if chunk_id not in all_results or score > all_results[chunk_id]["score"]:
                    all_results[chunk_id] = {
                        "id": chunk_id,
                        "document": results["documents"][0][j],
                        "metadata": results["metadatas"][0][j],
                        "score": score,
                    }

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
        # With cosine distance (0..2), map to [0..1] where 1 is best.
        # For older indexes / other metrics, fall back to a generic bounded transform.
        if distance <= 2.0:
            score = 1.0 - (distance / 2.0)
        else:
            score = 1.0 / (1.0 + distance)
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0

        formatted.append({
            "file_path": meta.get("file_path"),
            "start_line": meta.get("start_line"),
            "end_line": meta.get("end_line"),
            "content": results["documents"][0][i],
            "score": score,
            "symbols": symbols,
            "language": meta.get("language") or "",
            "last_modified": float(meta.get("last_modified") or 0.0),
        })

    return formatted


def clear_index(project_path: Path):
    """Clear index for project."""
    client = get_client()
    name = get_collection_name(project_path)
    try:
        client.delete_collection(name)
    except Exception:
        pass


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
