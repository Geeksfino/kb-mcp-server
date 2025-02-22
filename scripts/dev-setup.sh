#!/bin/bash

# Exit on error
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Setting up development environment..."

# Check if .env already exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo ".env file already exists. Skipping..."
    exit 0
fi

# Copy .env.example to .env
echo "Creating .env file from .env.example..."
cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"

# Detect if CUDA is available
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null; then
    echo "CUDA detected, enabling GPU support..."
    sed -i '' 's/TXTAI_MODEL_GPU=.*/TXTAI_MODEL_GPU=true/' "$PROJECT_ROOT/.env"
else
    echo "CUDA not detected, disabling GPU support..."
    sed -i '' 's/TXTAI_MODEL_GPU=.*/TXTAI_MODEL_GPU=false/' "$PROJECT_ROOT/.env"
fi

echo "Development environment setup complete!"
echo "You can customize your settings by editing .env"
