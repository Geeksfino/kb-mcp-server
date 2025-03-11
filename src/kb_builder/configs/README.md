# Knowledge Base Configuration Templates

This directory contains YAML configuration templates for building knowledge bases with txtai. These templates are used during the knowledge base building phase with the `kb_builder` tool.

## Configuration Types

### Storage and Backend Configurations

- `memory.yml` - In-memory vectors (no persistence, fastest for development)
- `sqlite-faiss.yml` - SQLite for content + FAISS for vectors (local file-based persistence)
- `postgres-pgvector.yml` - PostgreSQL + pgvector (production-ready with full persistence)

### Domain-Specific Configurations

- `base.yml` - Foundation template with common settings
- `technical_docs.yml` - Optimized for technical documentation and manuals
- `research_papers.yml` - Optimized for academic and scientific papers
- `code_repositories.yml` - Optimized for code documentation and repositories
- `general_knowledge.yml` - Optimized for encyclopedic and general information
- `data_science.yml` - Optimized for data science content (tutorials, guides)

## Usage

To use these templates, specify the path to the desired configuration file when building your knowledge base:

```bash
# Using a domain-specific configuration
python -m kb_builder build --config src/kb_builder/configs/technical_docs.yml --input /path/to/documents

# Using a storage-specific configuration
python -m kb_builder build --config src/kb_builder/configs/postgres-pgvector.yml --input /path/to/documents
```

## Customization

These templates provide a starting point. You may need to adjust parameters based on your specific content and query patterns. Key areas to customize:

1. **Path**: Update the `path` parameter to specify where your index will be stored
2. **Textractor**: Adjust text extraction parameters based on your content structure
3. **Graph**: Tune graph parameters to match the relationship density in your content
4. **Search**: Adjust search parameters to match your typical query patterns

## Parameter Explanations

### Storage and Backend
- `path`: Where to save the index
- `content.path`: Storage location for document content
- `embeddings.backend`: Vector storage backend (faiss, pgvector, etc.)
- `graph.backend`: Graph storage backend (sqlite, networkx, etc.)

### Text Extraction
- `paragraphs`: Extract by paragraphs (good for articles)
- `sentences`: Extract by sentences (more granular)
- `minlength`: Minimum text length to include
- `backend`: Text extraction backend ("text", "markdown", etc.)

### Embeddings
- `path`: Model path for embeddings
- `hybrid`: Enable hybrid search (combining semantic and keyword)
- `scoring.method`: Scoring method (bm25, tfidf, etc.)

### Graph
- `minscore`: Minimum similarity for creating graph edges
- `centrality`: Algorithm for node importance (pagerank, betweenness, etc.)
- `max_hops`: Maximum graph traversal distance
- `min_score`: Minimum score threshold for search results

## Additional Resources

- [txtai Configuration Guide](https://neuml.github.io/txtai/api/configuration)
- [Embeddings Configuration](https://neuml.github.io/txtai/embeddings/configuration)
- [Pipeline Configuration](https://neuml.github.io/txtai/pipeline)
- [Workflow Configuration](https://neuml.github.io/txtai/workflow)
- `weights`: Balance between keyword (bm25) and semantic (similarity) search

## Best Practices

1. Start with the template closest to your content domain
2. Run test queries and evaluate results
3. Adjust parameters incrementally based on results
4. Consider using the generic query enhancement in `cli.py` alongside these configurations
