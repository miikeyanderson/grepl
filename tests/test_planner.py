import unittest


from grepl.planner import analyze_query


class TestPlanner(unittest.TestCase):
    def test_quoted_string_prefers_grep_fixed(self):
        plan = analyze_query('"handleUserLogin"')
        self.assertTrue(plan.run_grep)
        self.assertFalse(plan.run_semantic)
        self.assertEqual(plan.mode, "exact")
        self.assertEqual(plan.grep_pattern, "handleUserLogin")
        self.assertTrue(plan.grep_fixed)

    def test_regex_metacharacters_prefers_grep(self):
        plan = analyze_query("Auth.*Error")
        self.assertTrue(plan.run_grep)
        self.assertFalse(plan.run_semantic)

    def test_natural_language_prefers_semantic(self):
        plan = analyze_query("where we validate tokens")
        self.assertTrue(plan.run_semantic)

    def test_mixed_query_is_hybrid(self):
        plan = analyze_query("update the AuthService logic")
        self.assertTrue(plan.run_grep)
        self.assertTrue(plan.run_semantic)
        self.assertEqual(plan.mode, "hybrid")

    def test_force_flags(self):
        plan = analyze_query("anything", grep_only=True)
        self.assertTrue(plan.run_grep)
        self.assertFalse(plan.run_semantic)
        self.assertEqual(plan.grep_pattern, "anything")
        self.assertTrue(plan.grep_fixed)

        plan2 = analyze_query("anything", semantic_only=True)
        self.assertFalse(plan2.run_grep)
        self.assertTrue(plan2.run_semantic)


if __name__ == "__main__":
    unittest.main()
