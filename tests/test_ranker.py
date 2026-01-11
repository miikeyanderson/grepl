import unittest


from grepl.ranker import Hit, compute_lexical_overlap, merge_results, rerank


class TestRanker(unittest.TestCase):
    def test_merge_overlap_creates_hybrid(self):
        grep = [
            Hit(
                source="grep",
                file_path="a.py",
                start_line=10,
                end_line=20,
                score=0.0,
                preview="x",
                symbols=[],
                grep_score=1.0,
                semantic_score=0.0,
            )
        ]
        sem = [
            Hit(
                source="semantic",
                file_path="a.py",
                start_line=18,
                end_line=30,
                score=0.0,
                preview="y",
                symbols=[],
                grep_score=0.0,
                semantic_score=0.7,
            )
        ]

        merged = merge_results(grep, sem, overlap_lines=3)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].source, "hybrid")
        self.assertEqual(merged[0].start_line, 10)
        self.assertEqual(merged[0].end_line, 30)

    def test_rerank_diversity_cap(self):
        hits = [
            Hit("grep", "a.py", 1, 2, 0.0, "", [], grep_score=1.0),
            Hit("grep", "a.py", 3, 4, 0.0, "", [], grep_score=1.0),
            Hit("grep", "a.py", 5, 6, 0.0, "", [], grep_score=1.0),
            Hit("grep", "a.py", 7, 8, 0.0, "", [], grep_score=1.0),
            Hit("grep", "b.py", 1, 2, 0.0, "", [], grep_score=1.0),
        ]
        ranked = rerank(hits, max_per_file=3)
        self.assertLessEqual(sum(1 for h in ranked if h.file_path == "a.py"), 3)

    def test_compute_lexical_overlap(self):
        overlap = compute_lexical_overlap(
            "find auth session token",
            "def handle_auth_session(token):\n    pass",
            ["handle_auth_session"],
        )
        self.assertGreaterEqual(overlap, 0.5)


if __name__ == "__main__":
    unittest.main()
