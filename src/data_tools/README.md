# Knowledge Base System

A comprehensive knowledge base system for ingesting documents, generating embeddings, and searching through documents using txtai's powerful capabilities.

## Overview

This system provides a command-line interface for building and searching a knowledge base:

1. **Document Processing**: Ingest various document types directly using txtai's built-in capabilities.
2. **Embedding Generation**: Process the documents into embeddings for similarity search.
3. **Graph-based Search**: Utilize txtai's graph capabilities for enhanced search results.
4. **Domain-specific Configurations**: Optimize search parameters for different content domains.
5. **Generic Query Enhancement**: Dynamically boost relevance based on query terms.

## Command-Line Interface

The system provides a command-line interface for building and searching the knowledge base:

### Building the Knowledge Base

```bash
python -m data_tools.cli build --config path/to/config.yml --input path/to/documents --recursive
```

### Searching the Knowledge Base

```bash
python -m data_tools.cli retrieve path/to/embeddings "your search query" --graph
```

## Domain-Specific Configuration Templates

The system includes a set of pre-configured templates optimized for different content domains:

- `base.yml`: Foundation with common settings
- `technical_docs.yml`: For technical documentation
- `research_papers.yml`: For academic/scientific papers
- `code_repositories.yml`: For code documentation
- `general_knowledge.yml`: For encyclopedic content
- `data_science.yml`: For data science content

### Using Configuration Templates

Use the configuration helper to work with templates:

```bash
# List available templates
python -m data_tools.configs.config_helper list

# View a template
python -m data_tools.configs.config_helper view data_science.yml

# Create a custom configuration
python -m data_tools.configs.config_helper create data_science.yml custom_config.yml --path .txtai/kb-custom --max-hops 1 --min-score 0.5
```

## Generic Query Enhancement

The system implements a generic query enhancement approach that:

1. Extracts key terms from the query
2. Boosts result scores based on term matches
3. Re-ranks results based on enhanced scores

This approach works across any domain without hard-coding specific keywords and adapts automatically to different query types.

## Architecture

The system is built on top of txtai and follows these principles:

1. Maximize leverage of txtai's built-in functionality
2. Avoid reinventing or overengineering solutions
3. Use txtai's API as designed rather than building parallel implementations

## Dependencies

- Python 3.8+
- txtai
- Various document processing libraries (depending on document types)

## Configuration

See the `configs` directory for example configurations and the configuration helper utility.
