#!/bin/bash
# Run all quality checks

set -e

echo "==================================="
echo "Running Code Quality Checks"
echo "==================================="

echo ""
echo "1/3: Formatting check (Black --check)..."
uv run black --check backend/

echo ""
echo "2/3: Linting (Ruff)..."
uv run ruff check backend/

echo ""
echo "3/3: Type checking (mypy)..."
uv run mypy backend/

echo ""
echo "==================================="
echo "All checks passed!"
echo "==================================="
