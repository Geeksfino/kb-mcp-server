# Configuration Examples

This directory contains example configurations for different txtai setups:

## 1. [memory.yml](memory.yml)
- In-memory vectors (no persistence)
- Fastest for development and testing
- All data is lost on restart
- No additional dependencies

## 2. [sqlite-faiss.yml](sqlite-faiss.yml)
- SQLite for content storage
- FAISS for vector storage
- Local file-based persistence
- Good for development with persistence
- Dependencies: `pip install txtai[ann]`

## 3. [postgres-pgvector.yml](postgres-pgvector.yml)
- PostgreSQL for content storage
- pgvector for vector storage
- Production-ready with full persistence
- Dependencies:
  ```bash
  pip install txtai[ann,database]
  # Also requires PostgreSQL with pgvector extension
  ```

## Usage

1. Copy the desired example:
```bash
cp config/examples/memory.yml config.yml
```

2. Edit as needed and set:
```bash
export TXTAI_YAML_CONFIG=config.yml
```

## Additional Resources

- [txtai Configuration Guide](https://neuml.github.io/txtai/api/configuration)
- [Embeddings Configuration](https://neuml.github.io/txtai/embeddings/configuration)
- [Pipeline Configuration](https://neuml.github.io/txtai/pipeline)
- [Workflow Configuration](https://neuml.github.io/txtai/workflow)
