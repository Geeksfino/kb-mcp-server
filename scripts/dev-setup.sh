#!/bin/bash
# Set up development environment

echo "Setting up development environment..."

# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks if .git directory exists
if [ -d ".git" ]; then
    echo "Installing pre-commit hooks..."
    pre-commit install
fi

echo "Development environment setup complete!"
