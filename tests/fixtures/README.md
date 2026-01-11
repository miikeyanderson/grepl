# Semantic benchmark fixtures

The semantic benchmark uses a small JSON file to validate that semantic-only search
returns expected files in the top-N results when the repo is indexed and embeddings
are available.

## Adding new cases

1. Edit `tests/fixtures/semantic_benchmark.json`.
2. Add a new entry under `cases`:
   - `query`: a natural-language query.
   - `expected_path`: the repo-relative file path that should appear in the top-N.
3. Keep queries focused on stable identifiers or concepts so results are resilient
   to code changes.
