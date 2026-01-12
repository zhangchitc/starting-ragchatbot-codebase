#!/bin/bash
# Run linting checks

echo "Running Ruff linter..."
uv run ruff check backend/
echo "Done!"
