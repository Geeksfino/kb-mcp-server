#!/bin/bash
# kb_build.sh - Simplified wrapper for kb_builder

# Default config template
CONFIG_TEMPLATE="base"

# Help text
show_help() {
    cat << EOF
Usage: $(basename "$0") INPUT_DIR [CONFIG] [EXTRA_ARGS...]

A simplified wrapper for kb_builder that builds a knowledge base from documents.

Arguments:
    INPUT_DIR         Directory containing documents to process
    CONFIG            Either:
                      1. A template name from src/kb_builder/configs/ (e.g., technical_docs)
                      2. A path to a custom YAML config file (e.g., /path/to/my_config.yml)
                      (default: base)
    EXTRA_ARGS        Any additional arguments to pass to kb_builder

Available templates:
$(ls -1 src/kb_builder/configs/*.yml 2>/dev/null | sed 's|src/kb_builder/configs/||' | sed 's|\.yml$||' | sed 's|^|    - |')

Examples:
    # Basic usage with default template
    $(basename "$0") /path/to/docs

    # Use technical_docs template
    $(basename "$0") /path/to/docs technical_docs

    # Use a custom config file
    $(basename "$0") /path/to/docs /path/to/my_config.yml

    # Pass additional arguments
    $(basename "$0") /path/to/docs technical_docs --update --export /path/to/export.tar.gz

    # Specify storage backend template
    $(basename "$0") /path/to/docs postgres-pgvector --update
EOF
}

# Check for help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Check for required arguments
if [ -z "$1" ]; then
    echo "Error: Input directory is required"
    show_help
    exit 1
fi

INPUT_DIR="$1"
shift

# Check for config parameter
CONFIG_PATH=""
if [ -n "$1" ] && [[ ! "$1" == --* ]]; then
    CONFIG_PARAM="$1"
    shift
    
    # Check if the parameter is a direct path to a YAML file
    if [[ "$CONFIG_PARAM" == *.yml ]] || [[ "$CONFIG_PARAM" == *.yaml ]]; then
        # If it's an absolute path or starts with ./ or ../
        if [[ "$CONFIG_PARAM" == /* ]] || [[ "$CONFIG_PARAM" == ./* ]] || [[ "$CONFIG_PARAM" == ../* ]]; then
            CONFIG_PATH="$CONFIG_PARAM"
            
            # Verify the file exists
            if [ ! -f "$CONFIG_PATH" ]; then
                echo "Error: Config file not found: $CONFIG_PATH"
                exit 1
            fi
        else
            # Assume it's a relative path from current directory
            CONFIG_PATH="$(pwd)/$CONFIG_PARAM"
            
            # Verify the file exists
            if [ ! -f "$CONFIG_PATH" ]; then
                echo "Error: Config file not found: $CONFIG_PATH"
                exit 1
            fi
        fi
    else
        # Assume it's a template name
        CONFIG_PATH="src/kb_builder/configs/${CONFIG_PARAM}.yml"
        
        # Verify the template exists
        if [ ! -f "$CONFIG_PATH" ]; then
            echo "Error: Config template '$CONFIG_PARAM' not found at $CONFIG_PATH"
            echo "Available templates:"
            ls -1 src/kb_builder/configs/*.yml | sed 's|src/kb_builder/configs/||' | sed 's|\.yml$||' | sed 's|^|    - |'
            exit 1
        fi
    fi
else
    # Use default template
    CONFIG_PATH="src/kb_builder/configs/${CONFIG_TEMPLATE}.yml"
    
    # Verify the template exists
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Error: Default config template '$CONFIG_TEMPLATE' not found at $CONFIG_PATH"
        echo "Available templates:"
        ls -1 src/kb_builder/configs/*.yml | sed 's|src/kb_builder/configs/||' | sed 's|\.yml$||' | sed 's|^|    - |'
        exit 1
    fi
fi

# Build the command
CMD="python -m src.kb_builder build --input \"$INPUT_DIR\" --config \"$CONFIG_PATH\""

# Add any remaining arguments
for arg in "$@"; do
    CMD+=" \"$arg\""
done

# Execute the command
echo "Running: $CMD"
eval "$CMD"
