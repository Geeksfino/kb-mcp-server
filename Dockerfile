# Use a prebuilt image that includes llama-cpp-python
FROM ghcr.io/allenporter/llama-cpp-server-cpu:v2.21.1

# Set working directory
WORKDIR /app

# Install additional system dependencies if needed
RUN apt-get update && \
    apt-get -y --no-install-recommends install \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install txtai with all components
RUN python -m pip install --no-cache-dir "txtai[all,pipeline,graph]>=8.3.1" && \
    python -c "import sys, importlib.util as util; 1 if util.find_spec('nltk') else sys.exit(); import nltk; nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger_eng'])"

# Copy project files
COPY . /app/

# Install MCP and other required dependencies
RUN pip install --no-cache-dir "mcp[cli]" trio httpx>=0.28.1 pydantic-settings>=2.0 \
    transformers>=4.30.0 sentence-transformers>=2.2.0 \
    datasets networkx>=2.8.0 matplotlib>=3.5.0 PyPDF2>=2.0.0 python-docx>=0.8.11 \
    beautifulsoup4>=4.10.0 pandas>=1.3.0 python-louvain>=0.16.0 markdown>=3.3.0

# Now install the project in development mode
RUN pip install --no-cache-dir -e .

# Make entrypoint script executable
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