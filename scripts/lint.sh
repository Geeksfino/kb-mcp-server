#!/bin/bash
# Run code formatting and linting

echo "Running isort..."
isort src/ tests/

echo "Running black..."
black src/ tests/

echo "Running mypy type checking..."
mypy src/

echo "All formatting and linting completed!"
