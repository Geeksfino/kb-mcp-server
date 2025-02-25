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
