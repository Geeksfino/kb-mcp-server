# Knowledge Base System

A comprehensive knowledge base system for ingesting documents, generating embeddings, building a knowledge graph, and searching through documents using various search strategies.

## Overview

This system provides a command-line interface for building and searching a knowledge base:

1. **Document Loading**: Ingest various document types (PDF, text, Markdown, HTML, Word) into a pipeline.
2. **Embedding Generation**: Process the documents into embeddings for similarity search.
3. **Knowledge Graph Construction**: Build a graph representation of document relationships.
4. **Search Interface**: Search the knowledge base using different strategies.

## Command-Line Interface

The system provides a command-line interface for building and searching the knowledge base:

### Building the Knowledge Base

```bash
python -m data_tools.cli build path/to/documents --recursive --build-graph --find-communities
```

Options:
- `--recursive`: Process directories recursively
- `--extensions`: File extensions to process (e.g., `--extensions pdf txt md`)
- `--output`: Output path for embeddings
- `--build-graph`: Build knowledge graph
- `--graph-path`: Path to save knowledge graph
- `--find-communities`: Find communities in knowledge graph
- `--config`: Path to configuration file

### Searching the Knowledge Base

```bash
python -m data_tools.cli search "your query" --search-type hybrid --limit 10
```

Options:
- `--search-type`: Type of search to perform:
  - `similar`: Semantic search using only dense vectors (embeddings)
  - `exact`: Keyword search using only sparse vectors (BM25)
  - `hybrid`: Combined search using both dense and sparse vectors (default)
  - `graph`: Graph-based search that finds related documents through connections
- `--limit`: Maximum number of results to return (default: 5)
- `--depth`: Maximum depth for graph traversal (for custom graph search, default: 2)
- `--graph-path`: Path to knowledge graph (for graph search)
- `--show-metadata`: Show document metadata
- `--extract-answers`: Extract answers using QA
- `--config`: Path to configuration file

### Example Usage

1. Build a knowledge base from a directory of documents:
   ```bash
   python -m data_tools.cli build ~/documents/research --recursive --extensions pdf txt md --build-graph
   ```

2. Search the knowledge base using hybrid search:
   ```bash
   python -m data_tools.cli search "What is machine learning?" --search-type hybrid --limit 5
   ```

3. Search using graph-based traversal:
   ```bash
   python -m data_tools.cli search "How does reinforcement learning work?" --search-type graph --depth 2
   ```

## Configuration

The system can be configured using a YAML file or environment variables:

```yaml
# Example configuration
path: ~/.txtai/embeddings
content: true
writable: true
embeddings:
  path: sentence-transformers/all-MiniLM-L6-v2
  storagepath: ~/.txtai/embeddings
  gpu: true
  normalize: true
  hybrid: true
  writable: true
graph:
  similarity: 0.75
  limit: 10
```

Environment variables:
- `KB_CONFIG`: Path to configuration file
- `KB_MODEL_PATH`: Path to embedding model
- `KB_INDEX_PATH`: Path to store embeddings
- `KB_MODEL_GPU`: Whether to use GPU for embeddings
- `KB_HYBRID_SEARCH`: Whether to enable hybrid search

## Components

The system consists of the following components:

1. **DocumentLoader**: Handles the ingestion of various document types
2. **DocumentProcessor**: Converts documents into embeddings for similarity search
3. **KnowledgeGraph**: Builds a graph representation of document relationships
4. **KnowledgeSearch**: Provides different search strategies against the knowledge base

These components are used internally by the CLI and can also be used programmatically if needed.
