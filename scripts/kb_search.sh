#!/bin/bash
# kb_search.sh - Simplified wrapper for kb_builder retrieve command

# Default values
LIMIT=5
USE_GRAPH=false
MIN_SIMILARITY=0.3

# Help text
show_help() {
    cat << EOF
Usage: $(basename "$0") EMBEDDINGS_PATH QUERY [OPTIONS]

A simplified wrapper for kb_builder that searches a knowledge base.

Required Arguments:
    EMBEDDINGS_PATH    Path to the embeddings database
    QUERY              Search query text

Options:
    --limit N          Maximum number of results to return (default: 5)
    --graph            Enable graph-enhanced search
    --min-similarity N Minimum similarity threshold (default: 0.3)

Examples:
    # Basic search
    $(basename "$0") /path/to/embeddings "What is machine learning?"

    # Search with graph enhancement
    $(basename "$0") /path/to/embeddings "What is machine learning?" --graph

    # Adjust result limit and similarity threshold
    $(basename "$0") /path/to/embeddings "What is machine learning?" --limit 10 --min-similarity 0.5
EOF
}

# Check for help
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Check for required arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Both embeddings path and query are required"
    show_help
    exit 1
fi

EMBEDDINGS_PATH="$1"
QUERY="$2"
shift 2

# Parse additional options
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --graph)
            USE_GRAPH=true
            shift
            ;;
        --min-similarity)
            MIN_SIMILARITY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build the command
CMD="python -m src.kb_builder retrieve \"$EMBEDDINGS_PATH\" \"$QUERY\" --limit $LIMIT --min_similarity $MIN_SIMILARITY"

# Add graph flag if enabled
if [ "$USE_GRAPH" = true ]; then
    CMD+=" --graph"
fi

# Execute the command
echo "Running: $CMD"
eval "$CMD"
