#!/bin/bash
# Auto-fix all code quality issues

set -e

echo "==================================="
echo "Auto-fixing Code Quality Issues"
echo "==================================="

echo ""
echo "1/2: Formatting with Black..."
uv run black backend/

echo ""
echo "2/2: Auto-fixing with Ruff..."
uv run ruff check --fix backend/

echo ""
echo "==================================="
echo "Done! Code has been formatted and auto-fixed."
echo "==================================="
