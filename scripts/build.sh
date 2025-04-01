#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Change to the project root directory
cd "$(dirname "$0")/.."

# Ensure we have the latest version of uv
pip install -U uv

# Clean up any previous build artifacts
rm -rf dist build *.egg-info

# Build the package using uv
echo "Building kb-mcp-server package..."
uv pip build

# Upload to PyPI
echo "Uploading to PyPI..."
uv pip publish

echo "Build and publish complete!"
echo "Package is now available at: https://pypi.org/project/kb-mcp-server/"
