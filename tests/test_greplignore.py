from pathlib import Path

from grepl.chunker import matches_pattern, should_index_file, should_skip_dir


def test_matches_directory_patterns_for_nested_paths():
    assert matches_pattern("node_modules/pkg/index.js", "node_modules/") is True
    assert matches_pattern("vendor/node_modules/pkg/index.js", "node_modules/") is True
    assert matches_pattern("build/output/app.js", "build/") is True
    assert matches_pattern("src/app.py", "node_modules/") is False


def test_should_skip_dir_respects_greplignore():
    root = Path("/repo")
    patterns = ["node_modules/", ".xcproject/"]

    assert should_skip_dir(root / "node_modules", patterns, root) is True
    assert should_skip_dir(root / "packages" / "node_modules", patterns, root) is True
    assert should_skip_dir(root / ".xcproject", patterns, root) is True
    assert should_skip_dir(root / "src", patterns, root) is False


def test_should_index_file_respects_directory_patterns():
    root = Path("/repo")
    patterns = ["node_modules/", "build/"]

    assert should_index_file(root / "node_modules" / "lib.js", patterns, root) is False
    assert should_index_file(root / "build" / "main.js", patterns, root) is False
    assert should_index_file(root / "src" / "app.js", patterns, root) is True
