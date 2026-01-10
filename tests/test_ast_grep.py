"""Tests for ast_grep module - uses fixtures to avoid sg dependency."""
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from grepl.ast_grep import (
    resolve_rule_file,
    list_available_rules,
    _parse_ast_grep_output,
)
from grepl.ranker import Hit


# Sample ast-grep JSON output fixture (matches sg 0.40+ format)
AST_GREP_FIXTURE = '''[
  {
    "text": "print(\\"Hello world\\")",
    "range": {
      "byteOffset": {"start": 100, "end": 125},
      "start": {"line": 5, "column": 8},
      "end": {"line": 5, "column": 30}
    },
    "file": "/test/file.swift",
    "lines": "        print(\\"Hello world\\")",
    "charCount": {"leading": 8, "trailing": 0},
    "language": "Swift",
    "metaVariables": {
      "single": {},
      "multi": {
        "ARGS": [
          {
            "text": "\\"Hello world\\"",
            "range": {
              "byteOffset": {"start": 106, "end": 119},
              "start": {"line": 5, "column": 14},
              "end": {"line": 5, "column": 27}
            }
          }
        ]
      },
      "transformed": {}
    }
  },
  {
    "text": "print(error.localizedDescription)",
    "range": {
      "byteOffset": {"start": 200, "end": 235},
      "start": {"line": 10, "column": 4},
      "end": {"line": 10, "column": 38}
    },
    "file": "/test/other.swift",
    "lines": "    print(error.localizedDescription)",
    "charCount": {"leading": 4, "trailing": 0},
    "language": "Swift",
    "metaVariables": {
      "single": {},
      "multi": {
        "ARGS": [
          {
            "text": "error.localizedDescription",
            "range": {
              "byteOffset": {"start": 206, "end": 232},
              "start": {"line": 10, "column": 10},
              "end": {"line": 10, "column": 36}
            }
          }
        ]
      },
      "transformed": {}
    }
  }
]'''


class TestAstGrepParsing(unittest.TestCase):
    """Test AST grep output parsing without requiring sg binary."""

    def test_parse_json_array_format(self):
        """Parse the JSON array format from sg --json."""
        hits = _parse_ast_grep_output(AST_GREP_FIXTURE, pattern="print($$$ARGS)")

        self.assertEqual(len(hits), 2)

        # First hit
        h1 = hits[0]
        self.assertEqual(h1.source, "ast")
        self.assertEqual(h1.file_path, "/test/file.swift")
        self.assertEqual(h1.start_line, 6)  # 0-indexed line 5 -> 1-indexed line 6
        self.assertEqual(h1.start_col, 8)
        self.assertEqual(h1.ast_score, 0.9)
        self.assertEqual(h1.ast_pattern, "print($$$ARGS)")
        self.assertIn("ARGS", h1.ast_captures)

        # Second hit
        h2 = hits[1]
        self.assertEqual(h2.file_path, "/test/other.swift")
        self.assertEqual(h2.start_line, 11)

    def test_parse_with_rule_file(self):
        """Parse output when using rule file."""
        hits = _parse_ast_grep_output(AST_GREP_FIXTURE, rule_file="swift-no-print.yml")

        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].ast_rule, "swift-no-print.yml")

    def test_parse_empty_output(self):
        """Empty output returns empty list."""
        hits = _parse_ast_grep_output("[]")
        self.assertEqual(hits, [])

    def test_parse_invalid_json(self):
        """Invalid JSON returns empty list (graceful failure)."""
        hits = _parse_ast_grep_output("not json at all")
        self.assertEqual(hits, [])

    def test_captures_multi_metavariables(self):
        """Multi metavariables ($$$) are captured correctly."""
        hits = _parse_ast_grep_output(AST_GREP_FIXTURE)

        h1 = hits[0]
        self.assertIn("ARGS", h1.ast_captures)
        self.assertEqual(h1.ast_captures["ARGS"], '"Hello world"')


class TestRuleResolution(unittest.TestCase):
    """Test rule file resolution."""

    def test_builtin_rule_resolution(self):
        """Built-in rules resolve correctly."""
        # This tests actual filesystem resolution
        rule_path, source = resolve_rule_file("swift-no-print")

        self.assertIsNotNone(rule_path)
        self.assertTrue(rule_path.exists())
        self.assertTrue(str(rule_path).endswith("swift-no-print.yml"))

    def test_nonexistent_rule_returns_none(self):
        """Non-existent rule returns None."""
        rule_path, source = resolve_rule_file("this-rule-does-not-exist-xyz")
        self.assertIsNone(rule_path)

    def test_list_available_rules(self):
        """List available rules finds rules."""
        available = list_available_rules()

        # Rules may be in 'user' or 'builtin' depending on install location
        all_rules = available.get("project", []) + available.get("user", []) + available.get("builtin", [])
        self.assertIn("swift-no-print", all_rules)


class TestExecutionPlan(unittest.TestCase):
    """Test ExecutionPlan serialization."""

    def test_to_dict_includes_ast_fields(self):
        """ExecutionPlan.to_dict() includes AST configuration."""
        from grepl.planner import QueryPlan, ExecutionPlan

        qp = QueryPlan(
            run_grep=False,
            run_semantic=False,
            run_ast=True,
            grep_pattern=None,
            grep_fixed=False,
            semantic_query=None,
            ast_patterns=("print($$$)",),
            ast_rules=(),
            ast_language="swift",
            ast_exhaustive=True,
            mode="exact",
            confidence=0.9,
        )

        ep = ExecutionPlan(
            query_plan=qp,
            query="",
            path=".",
            ast_top_files=50,
            ast_max_matches=200,
        )

        d = ep.to_dict()

        self.assertEqual(d["pipeline"], "ast(exhaustive)")
        self.assertIn("ast", d["stages"])
        self.assertEqual(d["stages"]["ast"]["patterns"], ["print($$$)"])
        self.assertEqual(d["stages"]["ast"]["language"], "swift")
        self.assertTrue(d["stages"]["ast"]["exhaustive"])

    def test_format_human_produces_output(self):
        """ExecutionPlan.format_human() produces non-empty output."""
        from grepl.planner import QueryPlan, ExecutionPlan

        qp = QueryPlan(
            run_grep=True,
            run_semantic=True,
            run_ast=True,
            grep_pattern="error",
            grep_fixed=False,
            semantic_query="error handling",
            ast_patterns=("try { $$ }",),
            ast_rules=(),
            ast_language=None,
            ast_exhaustive=False,
            mode="hybrid",
            confidence=0.5,
        )

        ep = ExecutionPlan(query_plan=qp, query="error handling", path=".")
        output = ep.format_human()

        self.assertIn("Pipeline:", output)
        self.assertIn("Semantic Stage:", output)
        self.assertIn("Grep Stage:", output)
        self.assertIn("AST Stage:", output)


if __name__ == "__main__":
    unittest.main()
