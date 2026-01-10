#!/bin/bash
# Comprehensive test script for all grepl features

echo "=========================================="
echo "Grepl Complete Feature Test"
echo "=========================================="
echo ""

echo "=== ENHANCED EXACT COMMAND ==="
echo ""
echo "1. Basic search with Rich Panels:"
echo "--------------------------------"
grepl exact "def" -p /Users/mikeyanderson/grepl/src/grepl -n 3
echo ""

echo "2. Case-insensitive search:"
echo "----------------------------"
grepl exact -i "error" -p /Users/mikeyanderson/grepl/src/grepl -n 3
echo ""

echo "3. All results from a pattern:"
echo "------------------------------"
grepl exact "import" -p /Users/mikeyanderson/grepl/src/grepl/cli.py
echo ""

echo "=== ENHANCED READ COMMAND ==="
echo ""
echo "4. Read with syntax highlighting:"
echo "--------------------------------"
grepl read /Users/mikeyanderson/grepl/src/grepl/cli.py:1-20
echo ""

echo "5. Read with line numbers:"
echo "-------------------------"
grepl read /Users/mikeyanderson/grepl/src/grepl/store.py:1-10
echo ""

echo "6. Read centered around a line:"
echo "-------------------------------"
grepl read /Users/mikeyanderson/grepl/src/grepl/chunker.py:45
echo ""

echo "=== JSON OUTPUT MODES ==="
echo ""
echo "7. Exact command - JSON:"
echo "-----------------------"
grepl exact "def" -p /Users/mikeyanderson/grepl/src/grepl -n 2 --json
echo ""

echo "8. Read command - JSON:"
echo "----------------------"
grepl read /Users/mikeyanderson/grepl/src/grepl/chunker.py:45-47 --json
echo ""

echo "=== STATUS COMMAND ==="
echo ""
echo "9. Check indexing status:"
echo "------------------------"
grepl status /Users/mikeyanderson/grepl
echo ""

echo "=== COMMAND HELP ==="
echo ""
echo "10. Exact command help:"
echo "---------------------"
grepl exact --help
echo ""

echo "=========================================="
echo "All Tests Complete!"
echo "=========================================="
