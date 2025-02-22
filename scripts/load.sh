#!/bin/bash

# Default values
INPUT_DIR=""
MODEL="sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE=2048
OVERLAP=20
BATCH_SIZE=32
FILE_TYPES=".txt,.md,.pdf,.doc,.docx"
USE_GPU=false
RECURSIVE=false
VERBOSE=false

# Database settings
DB_BACKEND="faiss"  # faiss, pgvector, sqlite-vec
DB_URL=""
CONTENT_URL=""

# Help text
show_help() {
    cat << EOF
Usage: $(basename "$0") [options]

Load documents into txtai embeddings database.

Required:
    -i, --input DIR         Input directory containing documents

Optional:
    -m, --model NAME        Model name (default: sentence-transformers/all-MiniLM-L6-v2)
    -c, --chunk-size NUM    Chunk size in characters (default: 2048)
    -o, --overlap NUM       Chunk overlap in characters (default: 20)
    -b, --batch-size NUM    Processing batch size (default: 32)
    -f, --file-types LIST   Comma-separated list of extensions (default: .txt,.md,.pdf,.doc,.docx)
    -g, --gpu              Use GPU acceleration
    -r, --recursive        Process subdirectories recursively
    -v, --verbose         Enable verbose logging

Database Options:
    --db-backend TYPE      Database backend: faiss, pgvector, sqlite-vec (default: faiss)
    --db-url URL          Database URL for pgvector
    --content-url URL     Database URL for content storage

Examples:
    # Basic usage with default settings
    $(basename "$0") -i /path/to/docs

    # Use GPU and process recursively
    $(basename "$0") -i /path/to/docs -g -r

    # Use PostgreSQL with pgvector
    $(basename "$0") -i /path/to/docs \\
      --db-backend pgvector \\
      --db-url "postgresql://user:pass@localhost:5432/dbname" \\
      --content-url "postgresql://user:pass@localhost:5432/dbname"
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -c|--chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        -o|--overlap)
            OVERLAP="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -f|--file-types)
            FILE_TYPES="$2"
            shift 2
            ;;
        -g|--gpu)
            USE_GPU=true
            shift
            ;;
        -r|--recursive)
            RECURSIVE=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --db-backend)
            DB_BACKEND="$2"
            shift 2
            ;;
        --db-url)
            DB_URL="$2"
            shift 2
            ;;
        --content-url)
            CONTENT_URL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_DIR" ]; then
    echo "Error: Input directory is required"
    show_help
    exit 1
fi

# Build command
CMD="python -m src.data_tools.document_loader"
CMD+=" --input \"$INPUT_DIR\""
CMD+=" --model \"$MODEL\""
CMD+=" --chunk-size $CHUNK_SIZE"
CMD+=" --overlap $OVERLAP"
CMD+=" --batch-size $BATCH_SIZE"
CMD+=" --file-types \"$FILE_TYPES\""

# Add optional flags
[ "$USE_GPU" = true ] && CMD+=" --gpu"
[ "$RECURSIVE" = true ] && CMD+=" --recursive"
[ "$VERBOSE" = true ] && CMD+=" --verbose"

# Add database options
CMD+=" --vector-backend \"$DB_BACKEND\""
[ -n "$DB_URL" ] && CMD+=" --vector-url \"$DB_URL\""
[ -n "$CONTENT_URL" ] && CMD+=" --content-url \"$CONTENT_URL\""

# Print command if verbose
[ "$VERBOSE" = true ] && echo "Running: $CMD"

# Execute command
eval "$CMD"
