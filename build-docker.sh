#!/bin/bash
# Script to build the Docker image with clean cache

# Default values
TRANSFORMERS_MODELS=""
SENTENCE_TRANSFORMERS_MODELS="sentence-transformers/nli-mpnet-base-v2"
HF_CACHE_DIR="$HOME/.cache/huggingface/hub"
IMAGE_NAME="txtai-mcp-server"
CLEAN_BUILD=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --transformers)
      TRANSFORMERS_MODELS="$2"
      shift 2
      ;;
    --sentence-transformers)
      SENTENCE_TRANSFORMERS_MODELS="$2"
      shift 2
      ;;
    --cache-dir)
      HF_CACHE_DIR="$2"
      shift 2
      ;;
    --image-name)
      IMAGE_NAME="$2"
      shift 2
      ;;
    --no-clean)
      CLEAN_BUILD=false
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build command
BUILD_CMD="docker build"

# Add no-cache option if clean build is requested
if [ "$CLEAN_BUILD" = true ]; then
  BUILD_CMD="$BUILD_CMD --no-cache"
fi

# Add build arguments
if [ -n "$TRANSFORMERS_MODELS" ]; then
  BUILD_CMD="$BUILD_CMD --build-arg HF_TRANSFORMERS_MODELS=\"$TRANSFORMERS_MODELS\""
fi

if [ -n "$SENTENCE_TRANSFORMERS_MODELS" ]; then
  BUILD_CMD="$BUILD_CMD --build-arg HF_SENTENCE_TRANSFORMERS_MODELS=\"$SENTENCE_TRANSFORMERS_MODELS\""
fi

if [ -n "$HF_CACHE_DIR" ]; then
  BUILD_CMD="$BUILD_CMD --build-arg HF_CACHE_DIR=\"$HF_CACHE_DIR\""
fi

# Add image name
BUILD_CMD="$BUILD_CMD -t $IMAGE_NAME ."

# Print the command
echo "Running: $BUILD_CMD"

# Execute the command
eval $BUILD_CMD
