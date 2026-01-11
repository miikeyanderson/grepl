from __future__ import annotations

import json
import os
from typing import Optional

import requests

from .embedder import OLLAMA_BASE_URL


DEFAULT_LLM_MODEL = os.getenv("GREPL_LLM_MODEL", "phi3")


class LLMClient:
    """Lightweight Ollama client for query expansion."""

    def __init__(self, model: str = DEFAULT_LLM_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return any(self.model in m for m in models)
        except requests.exceptions.RequestException:
            return False

    def generate(self, prompt: str, *, timeout: float = 20.0) -> Optional[str]:
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
        except requests.exceptions.RequestException:
            return None

    def expand_query(self, query: str) -> Optional[dict]:
        prompt = (
            "You are a code search assistant. Reformulate the query for code search. "
            "Return ONLY valid JSON with keys: reformulations (array of strings), "
            "code_patterns (array of strings), concepts (array of strings). "
            "Do not include extra text.\n\n"
            f"Query: {query}\n"
        )
        raw = self.generate(prompt)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
