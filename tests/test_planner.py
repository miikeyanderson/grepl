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

    def test_ast_patterns_enable_ast_stage(self):
        plan = analyze_query("error handling", ast_patterns=["try { $$ }"])
        self.assertTrue(plan.run_ast)
        self.assertEqual(plan.ast_patterns, ("try { $$ }",))
        self.assertFalse(plan.ast_exhaustive)

    def test_ast_rules_enable_ast_stage(self):
        plan = analyze_query("print", ast_rules=["swift-no-print"])
        self.assertTrue(plan.run_ast)
        self.assertEqual(plan.ast_rules, ("swift-no-print",))

    def test_ast_exhaustive_flag(self):
        plan = analyze_query("error", ast_patterns=["catch { $$ }"], ast_exhaustive=True)
        self.assertTrue(plan.run_ast)
        self.assertTrue(plan.ast_exhaustive)

    def test_codemod_strategy_forces_ast_exhaustive(self):
        plan = analyze_query("anything", ast_patterns=["print($$$)"], strategy="codemod")
        self.assertTrue(plan.run_ast)
        self.assertTrue(plan.ast_exhaustive)
        self.assertFalse(plan.run_grep)
        self.assertFalse(plan.run_semantic)

    def test_no_query_with_ast_runs_ast_only(self):
        """When no query but AST provided, run AST-only without grep/semantic."""
        plan = analyze_query("", ast_patterns=["print($$$)"])
        self.assertTrue(plan.run_ast)
        self.assertFalse(plan.run_grep)
        self.assertFalse(plan.run_semantic)
        self.assertTrue(plan.ast_exhaustive)  # Forced exhaustive when no narrowing

    def test_no_query_no_ast_returns_empty_plan(self):
        plan = analyze_query("")
        self.assertFalse(plan.run_ast)
        self.assertFalse(plan.run_grep)
        self.assertFalse(plan.run_semantic)


if __name__ == "__main__":
    unittest.main()
