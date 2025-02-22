#!/usr/bin/env bash

# Script to activate Python environment for project
# Usage: source scripts/activate.sh <conda_env_name>
# Example: source scripts/activate.sh 3.12

# Check if environment name is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide the conda environment name"
    echo "Usage: source scripts/activate.sh <conda_env_name>"
    echo "Example: source scripts/activate.sh 3.12"
    return 1
fi

CONDA_ENV_NAME=$1

# Get the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

# Check if conda is available
if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda command not found. Please install conda first."
    return 1
fi

# Check if uv is available
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv command not found. Please install uv first:"
    echo "pip install uv"
    return 1
fi

# Get conda root directory using json output
CONDA_ROOT=$(conda info --json 2>/dev/null | python -c "import sys, json; print(json.load(sys.stdin)['root_prefix'])")
CONDA_ENV_PATH="$CONDA_ROOT/envs/$CONDA_ENV_NAME"

# Check if environment exists
if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "Error: Conda environment not found at: $CONDA_ENV_PATH"
    echo "Available environments:"
    conda env list 2>/dev/null
    return 1
fi

# Check if python exists in the environment
if [ ! -x "$CONDA_ENV_PATH/bin/python" ]; then
    echo "Error: Python executable not found in environment: $CONDA_ENV_PATH/bin/python"
    return 1
fi

# Add conda environment bin to PATH
export PATH="$CONDA_ENV_PATH/bin:$PATH"

# Set environment variables
export CONDA_PREFIX="$CONDA_ENV_PATH"
export CONDA_DEFAULT_ENV="$CONDA_ENV_NAME"
export CONDA_PROMPT_MODIFIER="($CONDA_ENV_NAME) "

# Get Python version from conda environment
CONDA_PYTHON_VERSION=$("$CONDA_ENV_PATH/bin/python" --version 2>&1)
echo "Conda environment activated: $CONDA_ENV_NAME ($CONDA_PYTHON_VERSION)"

# Check if virtual environment exists and activate if needed
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # Store current Python path
    ORIGINAL_PYTHON_PATH=$(which python)
    
    # Activate virtual environment
    source "$PROJECT_ROOT/.venv/bin/activate"
    
    # Get virtual env Python version
    VENV_PYTHON_VERSION=$(python --version 2>&1)
    
    # Print environment info
    echo "Python environment activated:"
    echo "- Conda environment: $CONDA_ENV_NAME ($CONDA_PYTHON_VERSION)"
    echo "- Virtual environment: $PROJECT_ROOT/.venv ($VENV_PYTHON_VERSION)"
    
    # Check if versions match
    if [ "$CONDA_PYTHON_VERSION" != "$VENV_PYTHON_VERSION" ]; then
        echo "Warning: Python version mismatch between conda ($CONDA_PYTHON_VERSION) and virtualenv ($VENV_PYTHON_VERSION)"
        echo "To fix this, run:"
        echo "1. rm -rf $PROJECT_ROOT/.venv"
        echo "2. uv venv .venv"
        echo "3. uv sync"
        return 1
    fi

    # Check if pyproject.toml exists
    if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
        echo "Tip: Use 'uv sync' to install/update dependencies from pyproject.toml"
    fi
else
    echo "Virtual environment not found at $PROJECT_ROOT/.venv"
    echo "To create it, run:"
    echo "1. uv venv .venv"
    echo "2. uv sync"
fi