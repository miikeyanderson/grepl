import json
import unittest
from pathlib import Path

from grepl.store import check_semantic_ready, search


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "semantic_benchmark.json"


class TestSemanticBenchmark(unittest.TestCase):
    def test_semantic_benchmark_cases(self):
        ready, _ = check_semantic_ready(REPO_ROOT)
        if not ready:
            self.skipTest("Semantic search not ready (ollama/model/index missing)")

        data = json.loads(FIXTURE_PATH.read_text())
        top_k = int(data.get("top_k", 5))
        cases = data.get("cases", [])

        for case in cases:
            query = case["query"]
            expected_path = case["expected_path"]
            results = search(REPO_ROOT, query, limit=top_k, use_expansion=True)
            matches = [r for r in results if str(r.get("file_path", "")).endswith(expected_path)]
            self.assertTrue(
                matches,
                msg=f"Expected {expected_path} in top {top_k} for query '{query}'",
            )


if __name__ == "__main__":
    unittest.main()
