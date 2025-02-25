# Use Python 3.10 slim as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY main.py .
COPY README.md .

# Create volume mount points
RUN mkdir -p /data/embeddings

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Set environment variables
ENV TXTAI_STORAGE_MODE=persistence
ENV TXTAI_INDEX_PATH=/data/embeddings
ENV TXTAI_DATASET_ENABLED=true
ENV TXTAI_DATASET_NAME=web_questions
ENV TXTAI_DATASET_SPLIT=train

# Expose port for SSE
EXPOSE 8000

# Run MCP server with SSE transport
CMD ["mcp", "run", "--transport", "sse", "--host", "0.0.0.0", "--port", "8000", "main.py"]
