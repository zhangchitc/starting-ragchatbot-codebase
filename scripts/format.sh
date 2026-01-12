#!/bin/bash
# Format code with Black

echo "Formatting Python code with Black..."
uv run black backend/
echo "Done!"
