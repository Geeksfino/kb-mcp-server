# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install txtai and other dependencies
RUN uv pip install txtai[all,pipeline,graph]>=8.3.1 \
    trio httpx>=0.28.1 pydantic-settings>=2.0 \
    networkx>=2.8.0 matplotlib>=3.5.0 PyPDF2>=2.0.0 \
    python-docx>=0.8.11 python-louvain>=0.16.0 \
    fast-langdetect>=1.0.3 datasets torch>=2.0.0 \
    transformers>=4.30.0 sentence-transformers>=2.2.0 \
    bitsandbytes==0.42.0 beautifulsoup4>=4.10.0 \
    pandas>=1.3.0 markdown>=3.3.0

# Install MCP as a normal Python package
RUN uv pip install mcp==1.3.0

# Install the project
RUN uv pip install -e .

# Set Python path
ENV PYTHONPATH=/app

# Make scripts executable
RUN chmod +x /app/docker-entrypoint.sh
RUN chmod +x /app/download_models.py

# Create volume mount points
RUN mkdir -p /data/embeddings

# Download Hugging Face models if specified
ARG HF_TRANSFORMERS_MODELS=""
ARG HF_SENTENCE_TRANSFORMERS_MODELS=""
ARG HF_CACHE_DIR=""

# If HF_CACHE_DIR is provided, create a symbolic link to it
RUN if [ -n "$HF_CACHE_DIR" ] && [ -d "$HF_CACHE_DIR" ]; then \
    mkdir -p /root/.cache/huggingface && \
    ln -s "$HF_CACHE_DIR" /root/.cache/huggingface/hub; \
    fi

# Download models if specified
RUN if [ -n "$HF_TRANSFORMERS_MODELS" ] || [ -n "$HF_SENTENCE_TRANSFORMERS_MODELS" ]; then \
    python /app/download_models.py \
    --transformers "$HF_TRANSFORMERS_MODELS" \
    --sentence-transformers "$HF_SENTENCE_TRANSFORMERS_MODELS"; \
    fi

# Set environment variables
ENV TXTAI_STORAGE_MODE=persistence
ENV TXTAI_INDEX_PATH=/data/embeddings
ENV TXTAI_DATASET_ENABLED=true
ENV TXTAI_DATASET_NAME=web_questions
ENV TXTAI_DATASET_SPLIT=train

# Default environment variables for configuration
ENV PORT=8000
ENV HOST=0.0.0.0
ENV TRANSPORT=sse
ENV EMBEDDINGS_PATH=/data/embeddings
ENV HF_TRANSFORMERS_MODELS=$HF_TRANSFORMERS_MODELS
ENV HF_SENTENCE_TRANSFORMERS_MODELS=$HF_SENTENCE_TRANSFORMERS_MODELS

# Expose port
EXPOSE 8000

# Use the entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]