#!/bin/bash
# Build package distribution

echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

echo "Building package..."
python -m build

echo "Build complete! Distribution files are in dist/"
