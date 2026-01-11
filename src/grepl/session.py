from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SESSION_PATH = Path.home() / ".grepl" / "session.json"


@dataclass
class SessionState:
    current_file: Optional[str] = None
    cursor_line: Optional[int] = None
    open_files: List[str] = field(default_factory=list)
    last_query: Optional[str] = None
    last_results: List[Dict] = field(default_factory=list)
    last_search_id: Optional[str] = None

    def record_search(self, query: str, results: List[Dict]) -> None:
        self.last_query = query
        self.last_results = results
        self.last_search_id = str(int(time.time() * 1000))

    def update_focus(self, current_file: Optional[str], cursor_line: Optional[int]) -> None:
        if current_file:
            self.current_file = current_file
            if current_file not in self.open_files:
                self.open_files.append(current_file)
        if cursor_line is not None:
            self.cursor_line = cursor_line

    def match_read(self, file_path: str, line: int) -> Optional[int]:
        for idx, result in enumerate(self.last_results):
            if result.get("file_path") != file_path:
                continue
            start = int(result.get("start_line", 0))
            end = int(result.get("end_line", 0))
            if start <= line <= end:
                return idx
        return None


def load_session() -> SessionState:
    if not SESSION_PATH.exists():
        return SessionState()
    try:
        data = json.loads(SESSION_PATH.read_text())
        return SessionState(
            current_file=data.get("current_file"),
            cursor_line=data.get("cursor_line"),
            open_files=list(data.get("open_files", [])),
            last_query=data.get("last_query"),
            last_results=list(data.get("last_results", [])),
            last_search_id=data.get("last_search_id"),
        )
    except Exception:
        return SessionState()


def save_session(state: SessionState) -> None:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "current_file": state.current_file,
        "cursor_line": state.cursor_line,
        "open_files": state.open_files,
        "last_query": state.last_query,
        "last_results": state.last_results,
        "last_search_id": state.last_search_id,
    }
    SESSION_PATH.write_text(json.dumps(data, indent=2))


def update_session(
    *,
    current_file: Optional[str] = None,
    cursor_line: Optional[int] = None,
    last_query: Optional[str] = None,
) -> SessionState:
    state = load_session()
    state.update_focus(current_file, cursor_line)
    if last_query:
        state.last_query = last_query
    save_session(state)
    return state
