#!/bin/bash
# Run tests with coverage

echo "Running tests with coverage..."
pytest tests/ \
    --cov=src \
    --cov-report=term-missing \
    --cov-report=html:coverage \
    "$@"

echo "Test run complete! Check coverage/index.html for detailed coverage report."
