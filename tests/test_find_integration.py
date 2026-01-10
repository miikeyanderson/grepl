import json
import subprocess
import sys
import time
import unittest
from pathlib import Path


try:
    from grepl.embedder import check_ollama
    from grepl.store import has_index
except Exception:  # pragma: no cover
    check_ollama = None
    has_index = None


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = (REPO_ROOT / "src" / "grepl").resolve()


def _run_find(
    query: str,
    *,
    grep_only: bool = False,
    semantic_only: bool = False,
    precise: bool = False,
    top_k: int = 5,
    path: Path = TARGET,
):
    cmd = [
        sys.executable,
        "-m",
        "grepl.cli",
        "find",
        query,
        "-p",
        str(path),
        "-k",
        str(top_k),
        "--json",
    ]
    if grep_only:
        cmd.append("--grep-only")
    if semantic_only:
        cmd.append("--semantic-only")
    if precise:
        cmd.append("--precise")

    t0 = time.monotonic()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.monotonic() - t0

    out = p.stdout.strip()
    payload = json.loads(out) if out else {}
    return p.returncode, payload, dt, p.stderr


class TestFindIntegration(unittest.TestCase):
    def test_exact_identifier_grep_only(self):
        code, payload, _, _ = _run_find("chunk_codebase", grep_only=True)
        self.assertEqual(code, 0)
        self.assertEqual(payload.get("mode"), "exact")
        results = payload.get("results", [])
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all(r.get("source") == "grep" for r in results))

    def test_no_hits_returns_empty_in_json(self):
        code, payload, _, _ = _run_find("quantum computing logic", grep_only=True)
        self.assertEqual(code, 0)
        self.assertEqual(payload.get("results", []), [])

    def test_auto_mode_runs_and_returns_results(self):
        # This is intentionally a natural-language style query.
        code, payload, _, _ = _run_find("where we handle indexing")
        self.assertEqual(code, 0)
        self.assertGreaterEqual(len(payload.get("results", [])), 1)

    def test_semantic_expected_when_available(self):
        # If Ollama is running and this codebase has been indexed, auto mode should
        # return at least one result with a semantic component.
        if check_ollama is None or has_index is None:
            self.skipTest("grepl dependencies not importable")
        if not check_ollama():
            self.skipTest("Ollama not running")
        if not has_index(TARGET):
            self.skipTest("Target not indexed (run: grepl index src/grepl)")

        code, payload, _, _ = _run_find("how do we chunk the code into units")
        self.assertEqual(code, 0)
        results = payload.get("results", [])
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(any(float(r.get("semantic_score", 0.0)) > 0 for r in results))


if __name__ == "__main__":
    unittest.main()
