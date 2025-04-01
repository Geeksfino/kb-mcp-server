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

### Recommended: Using uv with Python 3.10+

We recommend using [uv](https://github.com/astral-sh/uv) with Python 3.10 or newer for the best experience. This provides better dependency management and ensures consistent behavior.

```bash
# Install uv if you don't have it already
pip install -U uv

# Create a virtual environment with Python 3.10 or newer
uv venv --python=3.10  # or 3.11, 3.12, etc.

# Activate the virtual environment (bash/zsh)
source .venv/bin/activate
# For fish shell
# source .venv/bin/activate.fish

# Install from PyPI
uv pip install kb-mcp-server
```

> **Note**: We pin transformers to version 4.49.0 to avoid deprecation warnings about `transformers.agents.tools` that appear in version 4.50.0 and newer. If you use a newer version of transformers, you may see these warnings, but they don't affect functionality.

### Using conda

```bash
# Create a new conda environment (optional)
conda create -n embedding-mcp python=3.10
conda activate embedding-mcp

# Install from PyPI
pip install kb-mcp-server
```

### From Source

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

# Option 1: Install from PyPI
uv pip install kb-mcp-server

# Option 2: Install from source (for development)
uv pip install -e .
```

### Using uvx (No Installation Required)

[uvx](https://github.com/astral-sh/uv) allows you to run packages directly from PyPI without installing them:

```bash
# Run the MCP server
uvx --from kb-mcp-server@0.3.0 kb-mcp-server --embeddings /path/to/knowledge_base

# Build a knowledge base
uvx --from kb-mcp-server@0.3.0 kb-build --input /path/to/documents --config config.yml

# Search a knowledge base
uvx --from kb-mcp-server@0.3.0 kb-search /path/to/knowledge_base "Your search query"
```

## Command Line Usage

### Building a Knowledge Base

You can use the command-line tools installed from PyPI, the Python module directly, or the convenient shell scripts:

#### Using the PyPI Installed Commands

```bash
# Build a knowledge base from documents
kb-build --input /path/to/documents --config config.yml

# Update an existing knowledge base with new documents
kb-build --input /path/to/new_documents --update

# Export a knowledge base for portability
kb-build --input /path/to/documents --export my_knowledge_base.tar.gz

# Search a knowledge base
kb-search /path/to/knowledge_base "What is machine learning?"

# Search with graph enhancement
kb-search /path/to/knowledge_base "What is machine learning?" --graph --limit 10
```

#### Using uvx (No Installation Required)

```bash
# Build a knowledge base from documents
uvx --from kb-mcp-server@0.3.0 kb-build --input /path/to/documents --config config.yml

# Update an existing knowledge base with new documents
uvx --from kb-mcp-server@0.3.0 kb-build --input /path/to/new_documents --update

# Export a knowledge base for portability
uvx --from kb-mcp-server@0.3.0 kb-build --input /path/to/documents --export my_knowledge_base.tar.gz

# Search a knowledge base
uvx --from kb-mcp-server@0.3.0 kb-search /path/to/knowledge_base "What is machine learning?"

# Search with graph enhancement
uvx --from kb-mcp-server@0.3.0 kb-search /path/to/knowledge_base "What is machine learning?" --graph --limit 10
```

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

#### Using the PyPI Installed Command

```bash
# Start with a specific knowledge base folder
kb-mcp-server --embeddings /path/to/knowledge_base_folder

# Start with a given knowledge base archive
kb-mcp-server --embeddings /path/to/knowledge_base.tar.gz
```

#### Using uvx (No Installation Required)

```bash
# Start with a specific knowledge base folder
uvx kb-mcp-server@0.2.6 --embeddings /path/to/knowledge_base_folder

# Start with a given knowledge base archive
uvx kb-mcp-server@0.2.6 --embeddings /path/to/knowledge_base.tar.gz
```

#### Using the Python Module

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
kb-mcp-server --embeddings /path/to/knowledge_base --host 0.0.0.0 --port 8000

# Or using uvx (no installation required)
uvx kb-mcp-server@0.2.6 --embeddings /path/to/knowledge_base --host 0.0.0.0 --port 8000

# Or using the Python module
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

## Configuring LLM Clients to Use the MCP Server

To configure an LLM client to use the MCP server, you need to create an MCP configuration file. Here's an example `mcp_config.json`:

### Using the server directly

If you use a virtual Python environment to install the server, you can use the following configuration - note that MCP host like Claude will not be able to connect to the server if you use a virtual environment, you need to use the absolute path to the Python executable of the virtual environment where you did "pip install" or "uv pip install", for example

```json
{
  "mcpServers": {
    "kb-server": {
      "command": "/your/home/project/.venv/bin/kb-mcp-server",
      "args": [
        "--embeddings", 
        "/path/to/knowledge_base.tar.gz"
      ],
      "cwd": "/path/to/working/directory"
    }
  }
}
```

### Using system default Python

If you use your system default Python, you can use the following configuration:

```json
{
    "rag-server": {
      "command": "python3",
      "args": [
        "-m",
        "txtai_mcp_server",
        "--embeddings",
        "/path/to/knowledge_base.tar.gz",
        "--enable-causal-boost"
      ],
      "cwd": "/path/to/working/directory"
    }
}
```

Alternatively, if you're using uvx, assuming you have uvx installed in your system via "brew install uvx" etc, or you 've installed uvx and made it globally accessible via:
```
# Create a symlink to /usr/local/bin (which is typically in the system PATH)
sudo ln -s /Users/cliang/.local/bin/uvx /usr/local/bin/uvx
```
This creates a symbolic link from your user-specific installation to a system-wide location. For macOS applications like Claude Desktop, you can modify the system-wide PATH by creating or editing a launchd configuration file:
```
# Create a plist file to set environment variables for all GUI applications
sudo nano /Library/LaunchAgents/environment.plist
```
Add this content:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>my.startup</string>
  <key>ProgramArguments</key>
  <array>
    <string>sh</string>
    <string>-c</string>
    <string>launchctl setenv PATH $PATH:/Users/cliang/.local/bin</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
</dict>
</plist>
```

Then load it:
```
sudo launchctl load -w /Library/LaunchAgents/environment.plist
```
You'll need to restart your computer for this to take effect, though.


```json
{
  "mcpServers": {
    "kb-server": {
      "command": "uvx",
      "args": [
        "kb-mcp-server@0.2.6",
        "--embeddings", "/path/to/knowledge_base",
        "--host", "localhost",
        "--port", "8000"
      ],
      "cwd": "/path/to/working/directory"
    }
  }
}
```

Place this configuration file in a location accessible to your LLM client and configure the client to use it. The exact configuration steps will depend on your specific LLM client.

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




