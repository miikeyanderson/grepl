import unittest

from grepl.store import apply_semantic_confidence_bounds, normalize_semantic_scores


class TestStoreSemanticScores(unittest.TestCase):
    def test_normalize_semantic_scores(self):
        scores = [0.2, 0.4, 0.6]
        normalized = normalize_semantic_scores(scores)
        self.assertAlmostEqual(normalized[0], 0.0)
        self.assertAlmostEqual(normalized[-1], 1.0)

    def test_apply_semantic_confidence_bounds(self):
        self.assertEqual(apply_semantic_confidence_bounds(0.1, floor=0.2), 0.0)
        self.assertEqual(apply_semantic_confidence_bounds(0.99, ceiling=0.95), 0.95)


if __name__ == "__main__":
    unittest.main()
