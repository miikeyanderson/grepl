#!/bin/bash
# Test script demonstrating grepl's enhanced CLI output

echo "=========================================="
echo "Grepl Enhanced CLI Output Demo"
echo "=========================================="
echo ""

# Test 1: Enhanced exact command with Rich Panels
echo "1. EXACT COMMAND: Search for 'def' in src/grepl (Rich Panels output)"
echo "-----------------------------------------------------------------"
grepl exact "def" -p /Users/mikeyanderson/grepl/src/grepl -n 5
echo ""

# Test 2: Read command with syntax highlighting
echo "2. READ COMMAND: View chunker.py with syntax highlighting"
echo "---------------------------------------------------------"
grepl read /Users/mikeyanderson/grepl/src/grepl/chunker.py:40-50
echo ""

# Test 3: JSON output for exact command
echo "3. JSON OUTPUT: Exact command results in JSON format"
echo "------------------------------------------------------"
grepl exact "def" -p /Users/mikeyanderson/grepl/src/grepl -n 2 --json
echo ""

# Test 4: JSON output for read command
echo "4. JSON OUTPUT: Read command results in JSON format"
echo "----------------------------------------------------"
grepl read /Users/mikeyanderson/grepl/src/grepl/chunker.py:45-47 --json
echo ""

# Test 5: Case-insensitive exact search
echo "5. CASE-INSENSITIVE: Search for 'error' (i flag)"
echo "-------------------------------------------------"
grepl exact -i "error" -p /Users/mikeyanderson/grepl/src/grepl -n 3
echo ""

# Test 6: Help command showing all options
echo "6. HELP: Show all available commands and options"
echo "------------------------------------------------"
grepl --help
echo ""

echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
