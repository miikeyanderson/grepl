"""Embeddings via multiple backends: Ollama (default) or OpenAI."""

import json
import os
import sqlite3
import threading
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from pathlib import Path
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

# Configuration
OLLAMA_BASE_URL = os.getenv("GREPL_OLLAMA_URL", "http://localhost:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Default models - prefer code-aware models
OLLAMA_DEFAULT_MODEL = os.getenv("GREPL_EMBED_MODEL", "mxbai-embed-large")
OPENAI_DEFAULT_MODEL = "text-embedding-3-small"

# Embedding dimensions by model (for zero-vector fallback)
MODEL_DIMENSIONS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "bge-m3": 1024,
    "all-minilm": 384,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

_CACHE_DIR = Path.home() / ".grepl" / "embedding_cache"
_CACHE_PATH = _CACHE_DIR / "embeddings.sqlite"


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension for this backend's model."""
        pass


class EmbeddingCache:
    """SQLite-backed embedding cache."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _CACHE_PATH
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    model TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (model, text_hash)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model)"
            )
            conn.commit()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_many(self, model: str, texts: List[str]) -> Dict[str, List[float]]:
        if not texts:
            return {}
        hashes = [self._hash_text(t) for t in texts]
        placeholders = ",".join("?" for _ in hashes)
        query = f"SELECT text_hash, embedding FROM embeddings WHERE model=? AND text_hash IN ({placeholders})"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, [model, *hashes]).fetchall()
        cached: Dict[str, List[float]] = {}
        for text_hash, embedding_json in rows:
            try:
                cached[text_hash] = json.loads(embedding_json)
            except Exception:
                continue
        return cached

    def set_many(self, model: str, items: Dict[str, List[float]]) -> None:
        if not items:
            return
        now = time.time()
        rows = [
            (model, text_hash, json.dumps(embedding), now)
            for text_hash, embedding in items.items()
        ]
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (model, text_hash, embedding, created_at) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()

    def warm(self, model: str, texts: List[str]) -> None:
        """Ensure embeddings are cached for the provided texts."""
        _ = get_embeddings(texts, model=model)


class OllamaBackend(EmbeddingBackend):
    """Ollama embedding backend."""

    def __init__(self, model: str = OLLAMA_DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        dim = self.get_dimension()

        for text in texts:
            try:
                truncated = text[:8000] if len(text) > 8000 else text
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": truncated},
                    timeout=60,
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
            except requests.exceptions.RequestException as exc:
                raise RuntimeError("Ollama embedding request failed") from exc

        return embeddings

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                return any(self.model in m for m in models)
            return False
        except requests.exceptions.RequestException:
            return False

    def get_dimension(self) -> int:
        return MODEL_DIMENSIONS.get(self.model, 768)


class OpenAIBackend(EmbeddingBackend):
    """OpenAI embedding backend."""

    def __init__(self, model: str = OPENAI_DEFAULT_MODEL, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or OPENAI_API_KEY

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise RuntimeError("OpenAI API key is not configured")

        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": [t[:8000] for t in texts],
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except requests.exceptions.RequestException as exc:
            raise RuntimeError("OpenAI embedding request failed") from exc

    def embed_single(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_dimension(self) -> int:
        return MODEL_DIMENSIONS.get(self.model, 1536)


# Global backend instance (lazy initialization)
_backend: Optional[EmbeddingBackend] = None
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get the embedding cache, initializing if needed."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache()
    return _embedding_cache


def get_backend() -> EmbeddingBackend:
    """Get the current embedding backend, initializing if needed."""
    global _backend
    if _backend is None:
        _backend = _select_backend()
    return _backend


def _select_backend() -> EmbeddingBackend:
    """Select the best available backend."""
    # Check for user preference in config
    config = load_config()
    preferred = config.get("preferred_backend")

    # Try user's preferred backend first
    if preferred == "openai" and OPENAI_API_KEY:
        backend = OpenAIBackend()
        if backend.is_available():
            return backend
    elif preferred == "ollama":
        return OllamaBackend()

    # Fallback to auto-selection
    # Check for OpenAI API key - use it if available
    if OPENAI_API_KEY:
        backend = OpenAIBackend()
        if backend.is_available():
            return backend

    # Default to Ollama
    return OllamaBackend()


def set_backend(backend: EmbeddingBackend) -> None:
    """Set a custom embedding backend."""
    global _backend
    _backend = backend


def get_embeddings(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """Get embeddings from the current backend.

    Args:
        texts: List of texts to embed
        model: Optional model override (ignored if backend already set)

    Returns:
        List of embedding vectors
    """
    backend = get_backend()
    cache = get_embedding_cache()

    # If model specified and different from backend, create new backend
    if model and isinstance(backend, OllamaBackend) and model != backend.model:
        backend = OllamaBackend(model=model)

    model_name = getattr(backend, "model", "default")
    cached = cache.get_many(model_name, texts)

    missing_texts: List[str] = []
    missing_hashes: List[str] = []
    for text in texts:
        text_hash = EmbeddingCache._hash_text(text)
        if text_hash not in cached:
            missing_texts.append(text)
            missing_hashes.append(text_hash)

    if missing_texts:
        batch_size = 8
        max_workers = min(8, (os.cpu_count() or 4))
        results: Dict[str, List[float]] = {}

        def _embed_batch(batch_texts: List[str]) -> List[List[float]]:
            return backend.embed(batch_texts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i in range(0, len(missing_texts), batch_size):
                batch = missing_texts[i:i + batch_size]
                future = executor.submit(_embed_batch, batch)
                futures[future] = (i, batch)

            for future in as_completed(futures):
                start_idx, batch = futures[future]
                try:
                    embeddings = future.result()
                except Exception:
                    embeddings = None
                # Skip caching failed or malformed batches; we'll fall back to
                # zero vectors at read time without poisoning the cache.
                if embeddings is None or len(embeddings) != len(batch):
                    continue
                dim = backend.get_dimension()
                for offset, embedding in enumerate(embeddings):
                    if not embedding or len(embedding) != dim:
                        continue
                    if not any(value != 0.0 for value in embedding):
                        continue
                    text_hash = missing_hashes[start_idx + offset]
                    results[text_hash] = embedding

        cache.set_many(model_name, results)
        cached.update(results)

    out: List[List[float]] = []
    for text in texts:
        text_hash = EmbeddingCache._hash_text(text)
        embedding = cached.get(text_hash)
        if embedding is None:
            embedding = [0.0] * backend.get_dimension()
        out.append(embedding)
    return out


def get_embedding(text: str, model: Optional[str] = None) -> List[float]:
    """Get single embedding from the current backend."""
    return get_embeddings([text], model)[0]


def check_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def check_model(model: str = OLLAMA_DEFAULT_MODEL) -> bool:
    """Check if the model is available in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m["name"] for m in response.json().get("models", [])]
            return any(model in m for m in models)
        return False
    except requests.exceptions.RequestException:
        return False


def get_available_models() -> List[str]:
    """List available Ollama models for embedding."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            return [m["name"] for m in response.json().get("models", [])]
        return []
    except requests.exceptions.RequestException:
        return []


def get_config_path() -> Path:
    """Get path to grepl config file."""
    config_dir = Path.home() / ".config" / "grepl"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.json"


def load_config() -> Dict:
    """Load grepl configuration."""
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_config(config: Dict) -> None:
    """Save grepl configuration."""
    config_path = get_config_path()
    with open(config_path, "w") as f:
        json.dump(config, indent=2, fp=f)


def get_backend_info() -> Dict:
    """Get information about the current backend."""
    backend = get_backend()
    backend_type = "openai" if isinstance(backend, OpenAIBackend) else "ollama"

    info = {
        "backend": backend_type,
        "model": backend.model,
        "dimensions": backend.get_dimension(),
        "available": backend.is_available(),
    }

    if backend_type == "ollama":
        info["url"] = backend.base_url
        info["ollama_running"] = check_ollama()

    return info


def list_available_backends() -> List[Dict]:
    """List all available embedding backends with their status."""
    backends = []

    # Check OpenAI
    openai_available = bool(OPENAI_API_KEY)
    if openai_available:
        try:
            backend = OpenAIBackend()
            openai_available = backend.is_available()
        except Exception:
            openai_available = False

    backends.append({
        "name": "openai",
        "model": OPENAI_DEFAULT_MODEL,
        "available": openai_available,
        "reason": "API key set" if OPENAI_API_KEY else "No API key (set OPENAI_API_KEY)",
    })

    # Check Ollama
    ollama_running = check_ollama()
    ollama_model_available = False
    if ollama_running:
        ollama_model_available = check_model(OLLAMA_DEFAULT_MODEL)

    backends.append({
        "name": "ollama",
        "model": OLLAMA_DEFAULT_MODEL,
        "available": ollama_running and ollama_model_available,
        "reason": (
            "Ready" if ollama_running and ollama_model_available
            else f"Model '{OLLAMA_DEFAULT_MODEL}' not found" if ollama_running
            else "Ollama not running"
        ),
    })

    return backends


def set_preferred_backend(backend_name: str) -> None:
    """Set the preferred embedding backend."""
    global _backend

    config = load_config()
    config["preferred_backend"] = backend_name
    save_config(config)

    # Reset backend to force re-selection
    _backend = None
