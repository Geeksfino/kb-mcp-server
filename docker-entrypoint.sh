#!/bin/bash
set -e

# Default values
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}
TRANSPORT=${TRANSPORT:-sse}
EMBEDDINGS_PATH=${EMBEDDINGS_PATH:-/data/embeddings}

# Handle embeddings path
if [ -n "$EMBEDDINGS_PATH" ]; then
  # Check if the embeddings path is a tar.gz file
  if [[ "$EMBEDDINGS_PATH" == *.tar.gz ]]; then
    echo "Detected tar.gz embeddings file: $EMBEDDINGS_PATH"
    # Pass the path directly to the embeddings parameter
    EMBEDDINGS_ARG="--embeddings $EMBEDDINGS_PATH"
  else
    echo "Using embeddings directory: $EMBEDDINGS_PATH"
    # Set environment variable for directory-based embeddings
    export TXTAI_EMBEDDINGS=$EMBEDDINGS_PATH
    EMBEDDINGS_ARG="--embeddings $EMBEDDINGS_PATH"
  fi
else
  EMBEDDINGS_ARG=""
fi

# Print configuration
echo "Starting TxtAI MCP Server with:"
echo "  - Transport: $TRANSPORT"
echo "  - Host: $HOST"
echo "  - Port: $PORT"
echo "  - Embeddings: $EMBEDDINGS_PATH"

# Run the server with the specified parameters
exec python -m txtai_mcp_server --transport "$TRANSPORT" --host "$HOST" --port "$PORT" $EMBEDDINGS_ARG
