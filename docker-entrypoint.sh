#!/bin/bash
set -e

# Default values
PORT=${PORT:-8000}
HOST=${HOST:-0.0.0.0}
TRANSPORT=${TRANSPORT:-sse}
EMBEDDINGS_PATH=${EMBEDDINGS_PATH:-/data/embeddings}
CONFIG_FILE=${CONFIG_FILE:-config.yml}

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

# Handle config file if it exists and is not empty
CONFIG_ARG=""
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
  echo "Using config file: $CONFIG_FILE"
  CONFIG_ARG="--config $CONFIG_FILE"
fi

# Print configuration
echo "Starting TxtAI MCP Server with:"
echo "  - Transport: $TRANSPORT"
echo "  - Host: $HOST"
echo "  - Port: $PORT"
echo "  - Embeddings: $EMBEDDINGS_PATH"
if [ -n "$CONFIG_ARG" ]; then
  echo "  - Config: $CONFIG_FILE"
fi

# Print pre-cached models if any
if [ -n "$HF_TRANSFORMERS_MODELS" ]; then
  echo "  - Pre-cached Transformer models: $HF_TRANSFORMERS_MODELS"
fi
if [ -n "$HF_SENTENCE_TRANSFORMERS_MODELS" ]; then
  echo "  - Pre-cached Sentence Transformer models: $HF_SENTENCE_TRANSFORMERS_MODELS"
fi

# Run the server with the specified parameters
exec python -m txtai_mcp_server --transport "$TRANSPORT" --host "$HOST" --port "$PORT" $EMBEDDINGS_ARG $CONFIG_ARG
