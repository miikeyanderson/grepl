"""Daemon server for in-memory code search."""

import hashlib
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Set

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from watchfiles import watch

from ..chunker import chunk_file
from ..embedder import get_embedding, get_embeddings
from ..session import load_session, save_session
from ..store import delete_chunks_for_file, get_collection, upsert_file_chunks, delete_code_graph_for_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    current_file: Optional[str] = None
    cursor_line: Optional[int] = None


class SearchResult(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float
    symbols: List[str]


class HealthResponse(BaseModel):
    status: str
    project_path: str
    chunks: int
    uptime: float


DIRTY_THRESHOLD = 20
DIRTY_IDLE_FLUSH_SECONDS = 2.0


class InMemoryIndex:
    """In-memory vector index for fast queries."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []
        self.documents: List[str] = []
        self.ids: List[str] = []
        self.dirty_files: Set[str] = set()
        self._reindex_lock = threading.Lock()
        self._last_dirty_at: Optional[float] = None

    def load(self):
        """Load embeddings from ChromaDB into memory."""
        logger.info(f"Loading index for {self.project_path}")
        collection = get_collection(self.project_path)

        data = collection.get(include=["embeddings", "documents", "metadatas"])

        if not data["ids"]:
            logger.warning("No embeddings found in collection")
            self.embeddings = np.array([])
            return

        self.ids = data["ids"]
        self.embeddings = np.array(data["embeddings"])
        self.documents = data["documents"]
        self.metadata = data["metadatas"]

        logger.info(f"Loaded {len(self.ids)} chunks into memory")

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search using in-memory cosine similarity."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        query_embedding = get_embedding(query)
        query_vec = np.array(query_embedding)

        # Normalize vectors for cosine similarity
        query_norm_val = np.linalg.norm(query_vec)
        if query_norm_val == 0:
            return []
        query_norm = query_vec / query_norm_val
        embeddings_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        embeddings_norms[embeddings_norms == 0] = 1.0
        embeddings_norm = self.embeddings / embeddings_norms

        # Compute cosine similarity
        similarities = np.dot(embeddings_norm, query_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:limit]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            meta = self.metadata[idx] or {}

            symbols = meta.get("symbols", "[]")
            if isinstance(symbols, str):
                try:
                    symbols = json.loads(symbols)
                except Exception:
                    symbols = []

            results.append(SearchResult(
                file_path=meta.get("file_path", ""),
                start_line=meta.get("start_line", 0),
                end_line=meta.get("end_line", 0),
                content=self.documents[idx],
                score=score,
                symbols=symbols or []
            ))

        return results

    def mark_dirty(self, file_path: str):
        """Mark file as dirty, trigger reindex if threshold reached."""
        should_reindex = False
        with self._reindex_lock:
            self.dirty_files.add(file_path)
            self._last_dirty_at = time.time()
            logger.debug(
                f"File dirty: {file_path} ({len(self.dirty_files)}/{DIRTY_THRESHOLD})"
            )
            should_reindex = len(self.dirty_files) >= DIRTY_THRESHOLD
        if should_reindex:
            self._do_reindex()

    def flush_if_idle(self, idle_seconds: float):
        """Flush dirty files after the index has been idle for a bit."""
        with self._reindex_lock:
            if not self.dirty_files or self._last_dirty_at is None:
                return
            idle_for = time.time() - self._last_dirty_at
        if idle_for >= idle_seconds:
            self._do_reindex()

    def flush(self):
        """Reindex any pending dirty files."""
        self._do_reindex()

    def _do_reindex(self):
        """Reindex dirty files and reload memory."""
        with self._reindex_lock:
            if not self.dirty_files:
                return
            dirty = self.dirty_files.copy()
            self.dirty_files.clear()
            self._last_dirty_at = None

        logger.info(f"Reindexing {len(dirty)} dirty files")

        for file_path in dirty:
            delete_chunks_for_file(self.project_path, file_path)
            delete_code_graph_for_file(file_path)
            path = Path(file_path)
            if path.exists():
                chunks = chunk_file(path)
                if chunks:
                    upsert_file_chunks(self.project_path, chunks)
                    logger.debug(f"Reindexed {file_path}: {len(chunks)} chunks")

        self.load()
        logger.info("Reindex complete")

    def update_file(self, file_path: str):
        """Handle file change by marking dirty (if not ignored)."""
        from ..chunker import load_greplignore, should_index_file

        path = Path(file_path)
        patterns = load_greplignore(self.project_path)

        if should_index_file(path, patterns, self.project_path):
            self.mark_dirty(file_path)


class GreplDaemon:
    """Main daemon process."""

    def __init__(self, project_path: Path, socket_path: Path):
        self.project_path = project_path.resolve()
        self.socket_path = socket_path
        self.index = InMemoryIndex(self.project_path)
        self.start_time = time.time()
        self.running = False
        self.change_queue: Queue = Queue()
        self.server: Optional[uvicorn.Server] = None

        self.app = FastAPI(title="Grepl Daemon")
        self._setup_routes()
        self._setup_signals()

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.post("/search", response_model=List[SearchResult])
        def search(req: SearchRequest):
            try:
                if req.current_file or req.cursor_line is not None:
                    state = load_session()
                    state.update_focus(req.current_file, req.cursor_line)
                    save_session(state)
                results = self.index.search(req.query, req.limit)
                return results
            except Exception as e:
                logger.error(f"Search error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health", response_model=HealthResponse)
        def health():
            return HealthResponse(
                status="ok",
                project_path=str(self.project_path),
                chunks=len(self.index.ids) if self.index.ids else 0,
                uptime=time.time() - self.start_time
            )

        @self.app.post("/shutdown")
        def shutdown():
            logger.info("Shutdown requested")
            self.stop()
            return {"status": "shutting down"}

    def _setup_signals(self):
        """Handle graceful shutdown on SIGTERM/SIGINT."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def _watch_files(self):
        """File watcher thread."""
        logger.info(f"Watching {self.project_path} for changes")

        try:
            for changes in watch(self.project_path):
                for change_type, path in changes:
                    self.change_queue.put((change_type, path))
        except Exception as e:
            logger.error(f"File watcher error: {e}")

    def _process_changes(self):
        """Process file change events."""
        while self.running:
            try:
                if not self.change_queue.empty():
                    change_type, path = self.change_queue.get(timeout=1)
                    self.index.update_file(str(path))
                else:
                    self.index.flush_if_idle(DIRTY_IDLE_FLUSH_SECONDS)
                    time.sleep(0.5)
            except Exception as e:
                logger.error(f"Change processor error: {e}")

    def start(self):
        """Start the daemon."""
        logger.info(f"Starting daemon for {self.project_path}")
        logger.info(f"Socket: {self.socket_path}")

        self.running = True

        # Pre-warm common query embeddings
        get_embeddings(["def", "class", "import"])

        # Load index into memory
        self.index.load()

        # Start file watcher thread
        watcher = threading.Thread(target=self._watch_files, daemon=True)
        watcher.start()

        # Start change processor thread
        processor = threading.Thread(target=self._process_changes, daemon=True)
        processor.start()

        # Ensure socket directory exists
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove old socket if exists
        if self.socket_path.exists():
            self.socket_path.unlink()

        # Start uvicorn server on Unix socket
        config = uvicorn.Config(
            self.app,
            uds=str(self.socket_path),
            log_level="info",
            access_log=False
        )
        self.server = uvicorn.Server(config)
        self.server.run()

    def stop(self):
        """Stop the daemon."""
        self.running = False
        self.index.flush()
        if self.server:
            self.server.should_exit = True
        if self.socket_path.exists():
            self.socket_path.unlink()


def get_daemon_socket_path(project_path: Path) -> Path:
    """Get socket path for a project."""
    path_hash = hashlib.md5(str(project_path.resolve()).encode()).hexdigest()[:8]
    cache_dir = Path.home() / ".cache" / "grepl"
    return cache_dir / f"daemon-{path_hash}.sock"


def start_daemon(project_path: Path):
    """Start daemon for a project."""
    socket_path = get_daemon_socket_path(project_path)
    daemon = GreplDaemon(project_path, socket_path)
    daemon.start()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m grepl.daemon.server <project_path>")
        sys.exit(1)

    project_path = Path(sys.argv[1])
    start_daemon(project_path)
