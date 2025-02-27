# Use Python 3.10 slim as base
FROM python:3.10-slim as builder

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
COPY README.md .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Second stage for a smaller final image
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files
COPY src/ src/
COPY README.md .
COPY docker-entrypoint.sh .

# Make entrypoint script executable
RUN chmod +x docker-entrypoint.sh

# Create volume mount points
RUN mkdir -p /data/embeddings

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

# Expose port (will be overridden by environment variable)
EXPOSE 8000

# Use the entrypoint script
ENTRYPOINT ["/app/docker-entrypoint.sh"]
