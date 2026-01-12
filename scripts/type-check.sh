#!/bin/bash
# Run type checking with mypy

echo "Running mypy type checker..."
uv run mypy backend/
echo "Done!"
