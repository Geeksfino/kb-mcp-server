#!/usr/bin/env fish

# Script to activate Python environment for project
# Usage: source scripts/activate.fish <conda_env_name>
# Example: source scripts/activate.fish 3.12

# Check if environment name is provided
if test (count $argv) -eq 0
    echo "Error: Please provide the conda environment name"
    echo "Usage: source scripts/activate.fish <conda_env_name>"
    echo "Example: source scripts/activate.fish 3.12"
    return 1
end

set -l CONDA_ENV_NAME $argv[1]

# Get the project root directory
set -l SCRIPT_DIR (dirname (status -f))
set -l PROJECT_ROOT (dirname $SCRIPT_DIR)

# Check if conda is available
if not type -q conda
    echo "Error: conda command not found. Please install conda first."
    return 1
end

# Check if uv is available
if not type -q uv
    echo "Error: uv command not found. Please install uv first:"
    echo "pip install uv"
    return 1
end

# Get conda root directory using json output
set -l CONDA_ROOT (conda info --json 2>/dev/null | python -c "import sys, json; print(json.load(sys.stdin)['root_prefix'])")
set -l CONDA_ENV_PATH "$CONDA_ROOT/envs/$CONDA_ENV_NAME"

# Check if environment exists
if not test -d "$CONDA_ENV_PATH"
    echo "Error: Conda environment not found at: $CONDA_ENV_PATH"
    echo "Available environments:"
    conda env list 2>/dev/null
    return 1
end

# Check if python exists in the environment
if not test -x "$CONDA_ENV_PATH/bin/python"
    echo "Error: Python executable not found in environment: $CONDA_ENV_PATH/bin/python"
    return 1
end

# Add conda environment bin to PATH
set -gx PATH "$CONDA_ENV_PATH/bin" $PATH

# Set environment variables
set -gx CONDA_PREFIX "$CONDA_ENV_PATH"
set -gx CONDA_DEFAULT_ENV "$CONDA_ENV_NAME"
set -gx CONDA_PROMPT_MODIFIER "($CONDA_ENV_NAME) "

# Get Python version from conda environment
set -l CONDA_PYTHON_VERSION ("$CONDA_ENV_PATH/bin/python" --version 2>&1)
echo "Conda environment activated: $CONDA_ENV_NAME ($CONDA_PYTHON_VERSION)"

# Check if virtual environment exists and activate if needed
if test -e $PROJECT_ROOT/.venv/bin/activate.fish
    # Store current Python path
    set -l ORIGINAL_PYTHON_PATH (which python)
    
    # Activate virtual environment
    source $PROJECT_ROOT/.venv/bin/activate.fish
    
    # Get virtual env Python version
    set -l VENV_PYTHON_VERSION (python --version 2>&1)
    
    # Print environment info
    echo "Python environment activated:"
    echo "- Conda environment: $CONDA_ENV_NAME ($CONDA_PYTHON_VERSION)"
    echo "- Virtual environment: $PROJECT_ROOT/.venv ($VENV_PYTHON_VERSION)"
    
    # Check if versions match
    if test "$CONDA_PYTHON_VERSION" != "$VENV_PYTHON_VERSION"
        echo "Warning: Python version mismatch between conda ($CONDA_PYTHON_VERSION) and virtualenv ($VENV_PYTHON_VERSION)"
        echo "To fix this, run:"
        echo "1. rm -rf $PROJECT_ROOT/.venv"
        echo "2. uv venv .venv"
        echo "3. uv sync"
        return 1
    end

    # Check if pyproject.toml exists
    if test -f "$PROJECT_ROOT/pyproject.toml"
        echo "Tip: Use 'uv sync' to install/update dependencies from pyproject.toml"
    end
else
    echo "Virtual environment not found at $PROJECT_ROOT/.venv"
    echo "To create it, run:"
    echo "1. uv venv .venv"
    echo "2. uv sync"
end
