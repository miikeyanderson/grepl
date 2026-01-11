from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


PROFILE_PATH = Path.home() / ".grepl" / "user_profile.json"


@dataclass
class UserProfile:
    preferred_languages: List[str] = field(default_factory=list)
    file_access_counts: Dict[str, int] = field(default_factory=dict)
    recent_queries: List[str] = field(default_factory=list)

    def record_file_access(self, file_path: str) -> None:
        self.file_access_counts[file_path] = self.file_access_counts.get(file_path, 0) + 1
        self._update_preferred_languages(file_path)

    def record_query(self, query: str, max_recent: int = 20) -> None:
        if not query:
            return
        if query in self.recent_queries:
            self.recent_queries.remove(query)
        self.recent_queries.append(query)
        self.recent_queries = self.recent_queries[-max_recent:]

    def total_accesses(self) -> int:
        return sum(self.file_access_counts.values())

    def affinity_for_file(self, file_path: str) -> float:
        total = self.total_accesses()
        if total <= 0:
            return 0.0
        return float(self.file_access_counts.get(file_path, 0)) / float(total)

    def _update_preferred_languages(self, file_path: str) -> None:
        ext = Path(file_path).suffix.lower()
        if not ext:
            return
        lang = _language_from_extension(ext)
        if not lang:
            return

        counts: Dict[str, int] = {}
        for fp, count in self.file_access_counts.items():
            ext_fp = Path(fp).suffix.lower()
            lang_fp = _language_from_extension(ext_fp)
            if lang_fp:
                counts[lang_fp] = counts.get(lang_fp, 0) + count

        sorted_langs = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        self.preferred_languages = [lang for lang, _ in sorted_langs[:5]]


def _language_from_extension(ext: str) -> str:
    mapping = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".swift": "swift",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".java": "java",
        ".kt": "kotlin",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".md": "markdown",
    }
    return mapping.get(ext, "")


def load_profile() -> UserProfile:
    if not PROFILE_PATH.exists():
        return UserProfile()
    try:
        data = json.loads(PROFILE_PATH.read_text())
        return UserProfile(
            preferred_languages=list(data.get("preferred_languages", [])),
            file_access_counts=dict(data.get("file_access_counts", {})),
            recent_queries=list(data.get("recent_queries", [])),
        )
    except Exception:
        return UserProfile()


def save_profile(profile: UserProfile) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "preferred_languages": profile.preferred_languages,
        "file_access_counts": profile.file_access_counts,
        "recent_queries": profile.recent_queries,
    }
    PROFILE_PATH.write_text(json.dumps(data, indent=2))


def record_file_access(file_path: str) -> None:
    profile = load_profile()
    profile.record_file_access(file_path)
    save_profile(profile)


def record_query(query: str) -> None:
    profile = load_profile()
    profile.record_query(query)
    save_profile(profile)
