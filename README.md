# Embedding MCP Server

A Model Context Protocol (MCP) server implementation powered by txtai, providing semantic search, knowledge graph capabilities, and AI-driven text processing through a standardized interface.

## The Power of txtai: All-in-one Embeddings Database

This project leverages [txtai](https://github.com/neuml/txtai), an all-in-one embeddings database for RAG leveraging semantic search, knowledge graph construction, and language model workflows. txtai offers several key advantages:

- **Unified Vector Database**: Combines vector indexes, graph networks, and relational databases in a single platform
- **Semantic Search**: Find information based on meaning, not just keywords
- **Knowledge Graph Integration**: Automatically build and query knowledge graphs from your data
- **Portable Knowledge Bases**: Save entire knowledge bases as compressed archives (.tar.gz) that can be easily shared and loaded
- **Extensible Pipeline System**: Process text, documents, audio, images, and video through a unified API
- **Local-first Architecture**: Run everything locally without sending data to external services

## How It Works

The project contains a knowledge base builder tool and a MCP server. The knowledge base builder tool is a command-line interface for creating and managing knowledge bases. The MCP server provides a standardized interface to access the knowledge base. 

It is not required to use the knowledge base builder tool to build a knowledge base. You can always build a knowledge base using txtai's programming interface by writing a Python script or even using a jupyter notebook. As long as the knowledge base is built using txtai, it can be loaded by the MCP server. Better yet, the knowledge base can be a folder on the file system or an exported .tar.gz file. Just give it to the MCP server and it will load it.

### 1. Build a Knowledge Base with kb_builder

The `kb_builder` module provides a command-line interface for creating and managing knowledge bases:

- Process documents from various sources (files, directories, JSON)
- Extract text and create embeddings
- Build knowledge graphs automatically
- Export portable knowledge bases

Note it is possibly limited in functionality and currently only provided for convenience.

### 2. Start the MCP Server

The MCP server provides a standardized interface to access the knowledge base:

- Semantic search capabilities
- Knowledge graph querying and visualization
- Text processing pipelines (summarization, extraction, etc.)
- Full compliance with the Model Context Protocol

## Installation

### Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n embedding-mcp python=3.10
conda activate embedding-mcp

# Clone the repository
git clone https://github.com/Geeksfino/kb-mcp-server.git.git
cd kb-mcp-server

# Install dependencies
pip install -e .
```

### Using uv (Faster Alternative)

```bash
# Install uv if not already installed
pip install uv

# Create a new virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

## Command Line Usage

### Building a Knowledge Base

You can use either the Python module directly or the convenient shell scripts:

#### Using the Python Module

```bash
# Build a knowledge base from documents
python -m kb_builder build --input /path/to/documents --config config.yml

# Update an existing knowledge base with new documents
python -m kb_builder build --input /path/to/new_documents --update

# Export a knowledge base for portability
python -m kb_builder build --input /path/to/documents --export my_knowledge_base.tar.gz
```

#### Using the Convenience Scripts

The repository includes convenient wrapper scripts that make it easier to build and search knowledge bases:

```bash
# Build a knowledge base using a template configuration
./scripts/kb_build.sh /path/to/documents technical_docs

# Build using a custom configuration file
./scripts/kb_build.sh /path/to/documents /path/to/my_config.yml

# Update an existing knowledge base
./scripts/kb_build.sh /path/to/documents technical_docs --update

# Search a knowledge base
./scripts/kb_search.sh /path/to/knowledge_base "What is machine learning?"

# Search with graph enhancement
./scripts/kb_search.sh /path/to/knowledge_base "What is machine learning?" --graph
```

Run `./scripts/kb_build.sh --help` or `./scripts/kb_search.sh --help` for more options.

### Starting the MCP Server

```bash
# Start with a specific knowledge base folder
python -m txtai_mcp_server --embeddings /path/to/knowledge_base_folder

# Start with a given knowledge base archive
python -m txtai_mcp_server --embeddings /path/to/knowledge_base.tar.gz
```
## MCP Server Configuration

The MCP server is configured using environment variables or command-line arguments, not YAML files. YAML files are only used for configuring txtai components during knowledge base building.

Here's how to configure the MCP server:

```bash
# Start the server with command-line arguments
python -m txtai_mcp_server --embeddings /path/to/knowledge_base --host 0.0.0.0 --port 8000

# Or use environment variables
export TXTAI_EMBEDDINGS=/path/to/knowledge_base
export MCP_SSE_HOST=0.0.0.0
export MCP_SSE_PORT=8000
python -m txtai_mcp_server
```

Common configuration options:
- `--embeddings`: Path to the knowledge base (required)
- `--host`: Host address to bind to (default: localhost)
- `--port`: Port to listen on (default: 8000)
- `--transport`: Transport to use, either 'sse' or 'stdio' (default: stdio)
- `--enable-causal-boost`: Enable causal boost feature for enhanced relevance scoring
- `--causal-config`: Path to custom causal boost configuration YAML file

## Advanced Knowledge Base Configuration

Building a knowledge base with txtai requires a YAML configuration file that controls various aspects of the embedding process. This configuration is used by the `kb_builder` tool, not the MCP server itself.

One may need to tune segmentation/chunking strategies, embedding models, and scoring methods, as well as configure graph construction, causal boosting, weights of hybrid search, and more.

Fortunately, txtai provides a powerful YAML configuration system that requires no coding. Here's an example of a comprehensive configuration for knowledge base building:

```yaml
# Path to save/load embeddings index
path: ~/.txtai/embeddings
writable: true

# Content storage in SQLite
content:
  path: sqlite:///~/.txtai/content.db

# Embeddings configuration
embeddings:
  # Model settings
  path: sentence-transformers/nli-mpnet-base-v2
  backend: faiss
  gpu: true
  batch: 32
  normalize: true
  
  # Scoring settings
  scoring: hybrid
  hybridalpha: 0.75

# Pipeline configuration
pipeline:
  workers: 2
  queue: 100
  timeout: 300

# Question-answering pipeline
extractor:
  path: distilbert-base-cased-distilled-squad
  maxlength: 512
  minscore: 0.3

# Graph configuration
graph:
  backend: sqlite
  path: ~/.txtai/graph.db
  similarity: 0.75  # Threshold for creating graph connections
  limit: 10  # Maximum connections per node
```

### Configuration Examples

The `src/kb_builder/configs` directory contains configuration templates for different use cases and storage backends:

#### Storage and Backend Configurations
- `memory.yml`: In-memory vectors (fastest for development, no persistence)
- `sqlite-faiss.yml`: SQLite for content + FAISS for vectors (local file-based persistence)
- `postgres-pgvector.yml`: PostgreSQL + pgvector (production-ready with full persistence)

#### Domain-Specific Configurations
- `base.yml`: Base configuration template
- `code_repositories.yml`: Optimized for code repositories
- `data_science.yml`: Configured for data science documents
- `general_knowledge.yml`: General purpose knowledge base
- `research_papers.yml`: Optimized for academic papers
- `technical_docs.yml`: Configured for technical documentation

You can use these as starting points for your own configurations:

```bash
python -m kb_builder build --input /path/to/documents --config src/kb_builder/configs/technical_docs.yml

# Or use a storage-specific configuration
python -m kb_builder build --input /path/to/documents --config src/kb_builder/configs/postgres-pgvector.yml
```

## Advanced Features

### Knowledge Graph Capabilities

The MCP server leverages txtai's built-in graph functionality to provide powerful knowledge graph capabilities:

- **Automatic Graph Construction**: Build knowledge graphs from your documents automatically
- **Graph Traversal**: Navigate through related concepts and documents
- **Path Finding**: Discover connections between different pieces of information
- **Community Detection**: Identify clusters of related information

### Causal Boosting Mechanism

The MCP server includes a sophisticated causal boosting mechanism that enhances search relevance by identifying and prioritizing causal relationships:

- **Pattern Recognition**: Detects causal language patterns in both queries and documents
- **Multilingual Support**: Automatically applies appropriate patterns based on detected query language
- **Configurable Boost Multipliers**: Different types of causal matches receive customizable boost factors
- **Enhanced Relevance**: Results that explain causal relationships are prioritized in search results

This mechanism significantly improves responses to "why" and "how" questions by surfacing content that explains relationships between concepts. The causal boosting configuration is highly customizable through YAML files, allowing adaptation to different domains and languages.


## License

MIT License - see LICENSE file for details




