from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


FEEDBACK_PATH = Path.home() / ".grepl" / "feedback.jsonl"
LTR_WEIGHTS_PATH = Path.home() / ".grepl" / "ltr_weights.json"

FEATURE_FIELDS = [
    "grep_score",
    "semantic_score",
    "ast_score",
    "symbol_boost",
    "lexical_boost",
    "recency_boost",
    "language_boost",
    "user_affinity_boost",
    "context_boost",
    "graph_boost",
]


@dataclass
class SearchEvent:
    query: str
    results: List[Dict]
    selected_idx: Optional[int]
    timestamp: float

    def to_json(self) -> str:
        return json.dumps({
            "query": self.query,
            "results": self.results,
            "selected_idx": self.selected_idx,
            "timestamp": self.timestamp,
        })


def log_search_event(event: SearchEvent) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(event.to_json() + "\n")


def load_events() -> List[SearchEvent]:
    if not FEEDBACK_PATH.exists():
        return []
    events = []
    with FEEDBACK_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                events.append(SearchEvent(
                    query=data.get("query", ""),
                    results=list(data.get("results", [])),
                    selected_idx=data.get("selected_idx"),
                    timestamp=float(data.get("timestamp", 0.0)),
                ))
            except Exception:
                continue
    return events


def _vectorize_result(result: Dict) -> List[float]:
    return [float(result.get(field, 0.0)) for field in FEATURE_FIELDS]


def train_ltr(min_events: int = 5) -> Tuple[bool, str]:
    events = load_events()
    if len(events) < min_events:
        return False, f"Not enough feedback events ({len(events)}/{min_events})"

    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:
        return False, f"scikit-learn not available: {exc}"

    X: List[List[float]] = []
    y: List[int] = []

    for event in events:
        if not event.results:
            continue
        selected = event.selected_idx
        for idx, result in enumerate(event.results):
            X.append(_vectorize_result(result))
            y.append(1 if selected is not None and idx == selected else 0)

    if not X:
        return False, "No training data found"

    model = LogisticRegression(max_iter=200, class_weight="balanced")
    model.fit(X, y)

    weights = {
        "intercept": float(model.intercept_[0]),
        "weights": {field: float(coef) for field, coef in zip(FEATURE_FIELDS, model.coef_[0])},
        "trained_at": time.time(),
    }
    LTR_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LTR_WEIGHTS_PATH.write_text(json.dumps(weights, indent=2))
    return True, f"Trained on {len(events)} events"


def load_ltr_weights() -> Optional[Dict[str, float]]:
    if not LTR_WEIGHTS_PATH.exists():
        return None
    try:
        data = json.loads(LTR_WEIGHTS_PATH.read_text())
        weights = data.get("weights", {})
        weights["intercept"] = float(data.get("intercept", 0.0))
        return weights
    except Exception:
        return None


def score_with_ltr(features: Dict[str, float], weights: Dict[str, float]) -> float:
    intercept = float(weights.get("intercept", 0.0))
    total = intercept
    for field in FEATURE_FIELDS:
        total += float(weights.get(field, 0.0)) * float(features.get(field, 0.0))
    # Sigmoid
    return 1.0 / (1.0 + pow(2.718281828459045, -total))
