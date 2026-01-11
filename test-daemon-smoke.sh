#!/bin/bash
set -euo pipefail

echo "🧪 Daemon Smoke Test"
echo "===================="
echo

# Use dev grepl
grepl_cmd() {
    PYTHONPATH=src /opt/homebrew/Cellar/grepl/0.2.12/libexec/bin/python3.11 -m grepl.cli "$@"
}

grepl_py() {
    PYTHONPATH=src /opt/homebrew/Cellar/grepl/0.2.12/libexec/bin/python3.11 "$@"
}

TEST_DIR="test_daemon_smoke"
TEST_ROOT="$(pwd)/${TEST_DIR}"
FILE_A="${TEST_ROOT}/file_a.py"
FILE_B="${TEST_ROOT}/file_b.py"
FILE_C="${TEST_ROOT}/file_c.py"

# Cleanup
echo "1. Cleanup..."
grepl_cmd daemon stop -p "${TEST_DIR}" 2>/dev/null || true
rm -rf "${TEST_DIR}"
mkdir -p "${TEST_DIR}"
grepl_cmd clear "${TEST_DIR}" >/dev/null 2>&1 || true

# Create initial files
echo "2. Creating 3 test files..."
echo "def test_a(): pass" > "${FILE_A}"
echo "def test_b(): pass" > "${FILE_B}"
echo "def test_c(): pass" > "${FILE_C}"

# Start daemon
echo "3. Starting daemon..."
grepl_cmd daemon start -p "${TEST_DIR}"
sleep 3

# Check status
echo "4. Checking daemon status..."
STATUS=$(grepl_cmd daemon status -p "${TEST_DIR}" 2>&1)
if echo "$STATUS" | grep -q "Running"; then
    echo "   ✓ Daemon is running"
else
    echo "   ✗ Daemon failed to start"
    exit 1
fi

# Edit 2 files (below threshold of 20)
echo "5. Editing 2 files (below 20 threshold)..."
echo "def test_a_modified(): pass" > "${FILE_A}"
echo "def test_b_modified(): pass" > "${FILE_B}"

# Wait for idle flush (DIRTY_IDLE_FLUSH_SECONDS = 2.0)
echo "6. Waiting 3s for idle flush..."
sleep 3

echo "   Verifying idle flush updated index..."
TEST_ROOT="${TEST_ROOT}" FILE_A="${FILE_A}" FILE_B="${FILE_B}" \
EXPECTED_A="test_a_modified" EXPECTED_B="test_b_modified" TIMEOUT_SECONDS="10" \
    grepl_py - <<'PY'
import os
import time
from pathlib import Path

from grepl.store import get_collection

project_path = Path(os.environ["TEST_ROOT"])
collection = get_collection(project_path)
deadline = time.time() + float(os.environ["TIMEOUT_SECONDS"])

checks = [
    (os.environ["FILE_A"], os.environ["EXPECTED_A"]),
    (os.environ["FILE_B"], os.environ["EXPECTED_B"]),
]

while True:
    missing = []
    for file_path, expected in checks:
        results = collection.get(where={"file_path": file_path}, include=["documents"])
        docs = results.get("documents") or []
        if not any(expected in (doc or "") for doc in docs):
            missing.append((file_path, expected))
    if not missing:
        print("   ✓ Idle flush reflected updated content")
        break
    if time.time() > deadline:
        raise SystemExit(
            "Timed out waiting for idle flush: "
            + ", ".join(f"{path} missing '{expected}'" for path, expected in missing)
        )
    time.sleep(0.5)
PY

# Create one more change to verify flush happened
echo "7. Making final change..."
echo "def test_c_modified(): pass" > "${FILE_C}"
sleep 3

# Stop daemon (should flush remaining)
echo "8. Stopping daemon (should flush remaining dirty files)..."
grepl_cmd daemon stop -p "${TEST_DIR}"

echo "   Verifying shutdown flush updated index..."
TEST_ROOT="${TEST_ROOT}" FILE_C="${FILE_C}" EXPECTED_C="test_c_modified" TIMEOUT_SECONDS="10" \
    grepl_py - <<'PY'
import os
import time
from pathlib import Path

from grepl.store import get_collection

project_path = Path(os.environ["TEST_ROOT"])
collection = get_collection(project_path)
deadline = time.time() + float(os.environ["TIMEOUT_SECONDS"])

file_path = os.environ["FILE_C"]
expected = os.environ["EXPECTED_C"]

while True:
    results = collection.get(where={"file_path": file_path}, include=["documents"])
    docs = results.get("documents") or []
    if any(expected in (doc or "") for doc in docs):
        print("   ✓ Shutdown flush reflected updated content")
        break
    if time.time() > deadline:
        raise SystemExit(f"Timed out waiting for shutdown flush for {file_path}")
    time.sleep(0.5)
PY

# Cleanup
echo "9. Cleanup..."
rm -rf "${TEST_DIR}"

echo
echo "✅ Smoke test passed!"
echo "   - Daemon started successfully"
echo "   - Handled <20 file changes with idle flush"
echo "   - Flushed on shutdown"
