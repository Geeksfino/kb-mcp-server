# TxtAI MCP Server

A Model Context Protocol (MCP) server implementation for txtai, providing semantic search, text processing, and AI capabilities through a standardized interface.

## Features

- **Semantic Search**: Search through documents and text using semantic understanding
- **Text Processing**: Summarization, translation, and text extraction
- **Model Management**: Access to various AI models and pipelines
- **MCP Compliance**: Full implementation of the Model Context Protocol

## Installation

### For Users

```bash
pip install txtai-mcp-server
```

### For Developers

1. Clone the repository:
```bash
git clone https://github.com/codeium/txtai-mcp-server.git
cd txtai-mcp-server
```

2. Set up development environment:
```bash
./scripts/dev-setup.sh
```

This will:
- Install the package in editable mode with development dependencies
- Set up pre-commit hooks
- Configure the development environment

## Development

### Project Structure

```
txtai-mcp-server/
├── src/                    # Source code
│   └── txtai_mcp_server/
│       ├── core/          # Core server implementation
│       ├── tools/         # MCP tools implementation
│       ├── resources/     # MCP resources implementation
│       └── prompts/       # MCP prompts implementation
├── tests/                 # Test suite
├── scripts/               # Development scripts
└── pyproject.toml        # Project configuration
```

### Development Scripts

- `./scripts/lint.sh`: Run code formatting and type checking
  - Runs isort for import sorting
  - Runs black for code formatting
  - Runs mypy for type checking

- `./scripts/test.sh`: Run tests with coverage
  - Runs pytest with coverage reporting
  - Generates HTML coverage report
  - Pass additional pytest arguments: `./scripts/test.sh -k test_name`

- `./scripts/build.sh`: Build package distribution
  - Cleans previous builds
  - Creates new distribution files

### Running Tests

```bash
# Run all tests
./scripts/test.sh

# Run specific test
./scripts/test.sh -k test_name

# Run tests with verbose output
./scripts/test.sh -v
```

### Code Quality

We use several tools to maintain code quality:

- **black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **pytest**: Testing framework
- **pre-commit**: Git hooks for code quality

Run all quality checks:
```bash
./scripts/lint.sh
```

## Configuration

The server supports two configuration methods:

### 1. YAML Configuration (Recommended)
Use txtai's native YAML configuration format for full access to all features:

1. Copy the example config:
```bash
cp config.example.yml config.yml
```

2. Edit `config.yml` to your needs and set:
```bash
export TXTAI_YAML_CONFIG=config.yml
```

See `config.example.yml` for a comprehensive example with all available options. For detailed documentation, see:
- [txtai Configuration Guide](https://neuml.github.io/txtai/api/configuration)
- [Embeddings Configuration](https://neuml.github.io/txtai/embeddings/configuration)
- [Pipeline Configuration](https://neuml.github.io/txtai/pipeline)
- [Workflow Configuration](https://neuml.github.io/txtai/workflow)

### 2. Environment Variables (Fallback)
For basic usage, configure through environment variables:

```bash
# Basic settings
export TXTAI_MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2
export TXTAI_STORAGE_MODE=memory  # or persistence
export TXTAI_INDEX_PATH=~/.txtai/embeddings
```

Or use a `.env` file (see `.env.example`).

### Configuration Priority

1. **YAML Configuration** (if `TXTAI_YAML_CONFIG` is set)
   - Full access to all txtai features
   - Native configuration format
   - Recommended for production use

2. **Environment Variables** (if no YAML config)
   - Basic configuration through `TXTAI_` prefixed variables
   - Limited to core settings
   - Good for development and testing

3. **Default Values** (if neither above is set)
   - Model: sentence-transformers/all-MiniLM-L6-v2
   - Storage: memory
   - Index: ~/.txtai/embeddings

### Examples

See `config.example.yml` for examples of:
1. Basic memory storage
2. Persistent storage with GPU
3. Full pipeline setup with QA
4. Workflow configuration
5. Cloud storage options

## Docker

### Running with Docker

The MCP server can be easily run using Docker. We use the official txtai image as a base:

```bash
# Build the Docker image (CPU version)
docker build -t txtai-mcp-server .

# Build with GPU support (requires modifying the FROM line in Dockerfile)
# FROM neuml/txtai-gpu:latest
docker build -t txtai-mcp-server-gpu .

# Build with pre-cached Hugging Face models
docker build \
  --build-arg HF_TRANSFORMERS_MODELS="bert-base-uncased,distilbert-base-uncased" \
  --build-arg HF_SENTENCE_TRANSFORMERS_MODELS="sentence-transformers/all-MiniLM-L6-v2" \
  -t txtai-mcp-server-with-models .

# Build with pre-cached models using the host's Hugging Face cache
# This will avoid downloading models that are already cached on the host
docker build \
  --build-arg HF_TRANSFORMERS_MODELS="bert-base-uncased,distilbert-base-uncased" \
  --build-arg HF_SENTENCE_TRANSFORMERS_MODELS="sentence-transformers/all-MiniLM-L6-v2" \
  --build-arg HF_CACHE_DIR="$HOME/.cache/huggingface/hub" \
  -t txtai-mcp-server-with-models .

# Run the container with default settings
docker run -p 8000:8000 txtai-mcp-server

# Run with custom port
docker run -p 9000:9000 -e PORT=9000 txtai-mcp-server

# Run with custom embeddings directory
docker run -p 8000:8000 -v /path/to/embeddings:/data/embeddings -e EMBEDDINGS_PATH=/data/embeddings txtai-mcp-server

# Run with embeddings tar.gz file
docker run -p 8000:8000 -v /path/to/embeddings.tar.gz:/data/embeddings.tar.gz -e EMBEDDINGS_PATH=/data/embeddings.tar.gz txtai-mcp-server
```

### Using Docker Compose

For a more convenient setup, use Docker Compose:

1. Create a `.env` file with your configuration (see `.env.example`)
2. Run the service:

```bash
# Start with default settings
docker-compose up

# Start with custom settings from .env file
docker-compose up

# Start with custom settings from command line
PORT=9000 EMBEDDINGS_PATH=/data/custom-embeddings docker-compose up

# Start with custom config file
LOCAL_CONFIG_PATH=./my_config.yml CONFIG_FILE=my_config.yml docker-compose up

# Start with pre-cached Hugging Face models
HF_TRANSFORMERS_MODELS="bert-base-uncased,roberta-base" \
HF_SENTENCE_TRANSFORMERS_MODELS="sentence-transformers/all-MiniLM-L6-v2" \
docker-compose up --build

# Start with pre-cached models using the host's Hugging Face cache
HF_TRANSFORMERS_MODELS="bert-base-uncased,roberta-base" \
HF_SENTENCE_TRANSFORMERS_MODELS="sentence-transformers/all-MiniLM-L6-v2" \
HF_CACHE_DIR="$HOME/.cache/huggingface/hub" \
docker-compose up --build
```

## Knowledge Management Tools

This project includes powerful data ingestion and knowledge graph tools for building AI applications with rich contextual understanding.

### Configuration

The data tools use txtai's Application and YAML configuration system. Create a `config.yaml` file:

```yaml
# Basic configuration
embeddings:
  path: ~/.txtai/embeddings
  content: true
  contentlength: 32768
  method: transformers
  transforms:
    - lowercase
    - strip
  score: bm25
  batch: 5000
  
# Vector embedding model  
pipeline:
  embedding:
    path: sentence-transformers/all-MiniLM-L6-v2

# Document ingestion
document:
  reader:
    path: txtai.reader.PDFReader
  splitter:
    path: txtai.splitter.Splitter
    params:
      chunking:
        size: 512
        overlap: 64

# Knowledge graph settings      
graph:
  similarity: 0.75
  limit: 10

# Optional NLP pipelines
pipelines:
  ner:
    path: txtai.pipeline.HFEntity
    params:
      model: dslim/bert-base-NER
```

### Document Ingestion

Ingest documents into the knowledge base:

```bash
# Process a directory of documents (recursive)
python -m src.data_tools.cli --config config.yaml ingest directory /path/to/documents --recursive

# Process a single file
python -m src.data_tools.cli --config config.yaml ingest file /path/to/document.pdf

# Process a HuggingFace dataset
python -m src.data_tools.cli --config config.yaml ingest dataset "dataset_name" --split train --text-field text
```

### Knowledge Graph

Build a knowledge graph from the ingested documents:

```bash
# Build a semantic knowledge graph
python -m src.data_tools.cli --config config.yaml graph --visualize

# Detect communities in the knowledge graph
python -m src.data_tools.cli --config config.yaml graph --detect-communities --visualize

# Customize graph parameters
python -m src.data_tools.cli --config config.yaml graph --min-similarity 0.8 --max-connections 15
```

### Semantic Search

Search documents with semantic understanding:

```bash
# Basic semantic search
python -m src.data_tools.cli --config config.yaml search "your search query"

# Graph-enhanced hybrid search
python -m src.data_tools.cli --config config.yaml search "your search query" --use-graph --context

# Customize search parameters
python -m src.data_tools.cli --config config.yaml search "your search query" --use-graph --graph-weight 0.7 --depth 2 --limit 10
```

### Programmatic Usage

You can also use the data tools in your own Python code:

```python
from data_tools.config import load_application
from data_tools.loader_utils import DocumentLoader
from data_tools.knowledge_graph import KnowledgeGraph

# Load application from config
app = load_application("config.yaml")

# Index documents
loader = DocumentLoader(app=app)
loader.process_directory("/path/to/documents", recursive=True)

# Build knowledge graph
kg = KnowledgeGraph(app=app)
kg.build_semantic_graph()

# Perform hybrid search
results = kg.hybrid_search("your search query", limit=5, graph_weight=0.6)

# Generate context for a query
context = kg.generate_context("your query", max_results=3, max_length=1000)
```

### Environment Variables

The Docker container supports the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| PORT | Server port | 8000 |
| HOST | Server host | 0.0.0.0 |
| TRANSPORT | Transport protocol (sse or stdio) | sse |
| EMBEDDINGS_PATH | Path to embeddings directory or tar.gz file | /data/embeddings |
| LOCAL_EMBEDDINGS_PATH | Local path to mount into container (docker-compose only) | ./embeddings |
| CONTAINER_EMBEDDINGS_PATH | Container path where embeddings are mounted (docker-compose only) | /data/embeddings |

## Knowledge Graph and RAG

The system now includes advanced knowledge graph and Retrieval Augmented Generation (RAG) capabilities.

### Knowledge Graph Features

- **Multiple Graph Building Approaches**:
  - **Semantic Graph Builder**: Creates connections based on semantic similarity
  - **Entity Graph Builder**: Extracts entities and relationships using LLMs
  - **Hybrid Graph Builder**: Combines both semantic and entity approaches

- **Graph Traversal and Querying**:
  - Path-based traversal with customizable hop limits
  - Cypher-like query language for complex path expressions
  - Community detection for identifying related content clusters

- **Visualization**:
  - Static graph visualization with multiple layout algorithms
  - Interactive HTML-based visualizations
  - Path and community visualization

### RAG Pipeline Features

- **Flexible Retrieval Methods**:
  - **Vector Retrieval**: Traditional embedding-based similarity search
  - **Graph Retrieval**: Retrieval based on graph relationships
  - **Path Retrieval**: Advanced retrieval using graph traversal paths

- **Citation Generation**:
  - Automatic citation of sources used in generation
  - Verification of generated content against source material
  - Coverage metrics for response verification

### CLI Usage Examples

#### Building a Knowledge Graph

```bash
# Build a semantic graph from documents in a directory
kb graph-build --input /path/to/documents --type semantic --output graph.pkl

# Build an entity-based graph with custom settings
kb graph-build --input /path/to/documents --type entity --similarity 0.8 --max-connections 10 --model "llm-model-name"

# Visualize the graph after building
kb graph-build --input /path/to/documents --visualize graph.html
```

#### Traversing a Knowledge Graph

```bash
# Simple traversal with a query
kb graph-traverse "your query" --input graph.pkl

# Advanced traversal with a path query
kb graph-traverse "your query" --path-query "(a)-[r]->(b)" --max-hops 3 --limit 5

# Visualize traversal results
kb graph-traverse "your query" --visualize path.html
```

#### Visualizing a Knowledge Graph

```bash
# Create an interactive HTML visualization
kb graph-visualize --input graph.pkl --output graph.html

# Create a static visualization with custom layout
kb graph-visualize --input graph.pkl --output graph.png --layout kamada_kawai --node-size 500 --font-size 10

# Visualize with community detection
kb graph-visualize --input graph.pkl --output communities.html --community-detection
```

#### Using the RAG Pipeline

```bash
# Simple RAG with vector retrieval
kb rag "your question" --retriever vector --model "llm-model-name"

# Graph-based RAG
kb rag "your question" --retriever graph --graph graph.pkl

# Path-based RAG with custom path expression
kb rag "your question" --retriever path --graph graph.pkl --path-expression "(a)-[r]->(b)"

# RAG with citations
kb rag "your question" --retriever vector --citations
```

### Python API Examples

```python
from data_tools.graph_builder import create_graph_builder
from data_tools.graph_traversal import GraphTraversal, PathQuery
from data_tools.rag import RAGPipeline
from data_tools.visualization import GraphVisualizer, VisualizationOptions

# Create a graph builder
builder = create_graph_builder("semantic", embeddings, {
    "similarity_threshold": 0.75,
    "max_connections": 5
})

# Build a graph
graph = builder.build(documents)
builder.save("graph.pkl")

# Create a graph traversal
traversal = GraphTraversal(graph, {
    "max_path_length": 3,
    "max_paths": 10
})

# Query paths
paths = traversal.query_paths("your query")

# Create a path query
query = PathQuery("(a)-[r]->(b)")
paths = traversal.query_paths("your query", query)

# Create a RAG pipeline
pipeline = RAGPipeline(embeddings, {
    "retriever": "vector",
    "generator": {
        "model": "model-name",
        "max_tokens": 1024
    },
    "citation": {
        "enabled": True
    }
})

# Generate a response
response = pipeline.generate("your question")

# Generate a response with citations
result = pipeline.generate_with_citations("your question")
print(result["response"])
print(result["citations"])
print(result["verification"])

# Visualize a graph
visualizer = GraphVisualizer(graph, VisualizationOptions({
    "layout": "spring",
    "node_size": 750,
    "font_size": 8
}))

# Create static visualization
visualizer.visualize_graph("graph.png")

# Create interactive HTML visualization
visualizer.export_to_html("graph.html")

# Visualize a path
visualizer.visualize_path(paths[0], "path.png")
```

## Usage

### Starting the Server

```python
from src.txtai_mcp_server.core import create_server

# Create server instance
mcp = create_server()

# Run with stdio transport
async with stdio_server() as (read_stream, write_stream):
    await mcp.run(read_stream, write_stream)
```

### Available Tools

- **Search Tools**
  - `semantic_search`: Search for semantically similar content
  - `add_content`: Add content to the search index
  - `delete_content`: Remove content from the index

- **Text Processing Tools**
  - `summarize`: Generate text summaries
  - `translate`: Translate text between languages
  - `extract_text`: Extract text from various formats

### Available Resources

- **Configuration Resources**
  - `config://embeddings`: Embeddings configuration
  - `config://pipelines`: Pipeline configurations
  - `config://server`: Server configuration

- **Model Resources**
  - `model://embeddings/{name}`: Embedding model information
  - `model://pipeline/{name}`: Pipeline information
  - `model://capabilities`: Available model capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - see LICENSE file for details
