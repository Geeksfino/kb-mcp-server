# Knowledge Graph and RAG System Design

This document outlines the comprehensive design for enhancing the `data_tools` module with advanced knowledge graph and Retrieval Augmented Generation (RAG) capabilities, based on the analysis of txtai example notebooks 55, 57, and 58.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Component Specifications](#component-specifications)
   - [Graph Building](#graph-building)
   - [Graph Traversal](#graph-traversal)
   - [RAG Pipeline](#rag-pipeline)
   - [Visualization](#visualization)
4. [Data Flow](#data-flow)
5. [Configuration](#configuration)
6. [CLI Interface](#cli-interface)
7. [Implementation Roadmap](#implementation-roadmap)

## System Overview

The enhanced system will provide three complementary approaches to knowledge graph creation and utilization:

1. **Semantic Graph Building** (Notebook 55): Automatic creation of graphs based on semantic similarity between document chunks
2. **Entity-Relationship Extraction** (Notebook 57): LLM-driven extraction of entities and their relationships from text
3. **Graph Path Traversal** (Notebook 58): Advanced graph traversal for comprehensive context building in RAG applications

These capabilities will be integrated into a flexible pipeline architecture that allows users to mix and match different approaches based on their specific needs.

## Architecture

The system follows a modular architecture with the following key components:

```
data_tools/
├── __init__.py
├── cli.py                  # Command-line interface
├── config.py               # Configuration handling
├── loader.py               # Document loading and processing
├── processor.py            # Document processing
├── kg.py                   # Core knowledge graph functionality
├── graph_builder.py        # Graph building strategies
├── graph_traversal.py      # Graph traversal and querying
├── rag.py                  # RAG pipeline implementation
├── visualization.py        # Graph visualization tools
└── search.py               # Search functionality
```

### Component Interactions

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Loader    │────▶│  Processor  │────▶│ Embeddings  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                                                ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    RAG      │◀───▶│   Graph     │◀───▶│   Graph     │
│   Pipeline  │     │  Traversal  │     │   Builder   │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       │                   ▼                   │
       │            ┌─────────────┐           │
       └───────────▶│Visualization│◀──────────┘
                    └─────────────┘
```

## Component Specifications

### Graph Building

#### `graph_builder.py`

This module will implement different strategies for building knowledge graphs:

1. **`GraphBuilder` (Abstract Base Class)**
   - Defines the interface for all graph building strategies
   - Methods:
     - `build(documents)`: Build a graph from documents
     - `save(path)`: Save the graph to a file
     - `load(path)`: Load a graph from a file

2. **`SemanticGraphBuilder`**
   - Builds graphs based on semantic similarity (Notebook 55 approach)
   - Configuration:
     - `similarity_threshold`: Minimum similarity score for creating edges
     - `max_connections`: Maximum number of connections per node
     - `bidirectional`: Whether to create bidirectional edges

3. **`EntityGraphBuilder`**
   - Builds graphs by extracting entities and relationships using LLMs (Notebook 57 approach)
   - Configuration:
     - `llm_model`: Model to use for entity extraction
     - `relationship_types`: Predefined relationship types to extract
     - `confidence_threshold`: Minimum confidence for extracted entities

4. **`HybridGraphBuilder`**
   - Combines semantic similarity and entity extraction approaches
   - Methods:
     - `filter_by_similarity(documents, query, limit)`: Filter documents by similarity
     - `extract_entities(documents)`: Extract entities from filtered documents
     - `merge_graphs(semantic_graph, entity_graph)`: Merge the two graph types

### Graph Traversal

#### `graph_traversal.py`

This module will implement advanced graph traversal capabilities:

1. **`GraphTraversal`**
   - Provides path-based traversal of knowledge graphs (Notebook 58 approach)
   - Methods:
     - `query_paths(query, path_expression)`: Query the graph using path expressions
     - `find_paths(start_node, end_node, max_hops)`: Find paths between nodes
     - `centrality_analysis()`: Identify central nodes in the graph
     - `community_detection()`: Detect communities in the graph

2. **`PathQuery`**
   - Parser and executor for Cypher-like path queries
   - Methods:
     - `parse(query_string)`: Parse a Cypher-like query
     - `execute(graph)`: Execute the query on a graph
     - `to_string()`: Convert the query back to a string

### RAG Pipeline

#### `rag.py`

This module will implement a flexible RAG pipeline:

1. **`RAGPipeline`**
   - Coordinates the RAG process
   - Methods:
     - `retrieve(query, **kwargs)`: Retrieve relevant context
     - `generate(query, context)`: Generate a response
     - `generate_with_citations(query, **kwargs)`: Generate a response with citations
     - `evaluate(query, response, ground_truth)`: Evaluate the quality of a response

2. **`Retriever` (Abstract Base Class)**
   - Defines the interface for context retrieval
   - Subclasses:
     - `VectorRetriever`: Simple vector similarity retrieval
     - `GraphRetriever`: Graph-based retrieval
     - `PathRetriever`: Path-based traversal retrieval

3. **`Generator`**
   - Handles LLM-based text generation
   - Configuration:
     - `model`: LLM model to use
     - `template`: Prompt template
     - `max_tokens`: Maximum tokens to generate

4. **`Citation`**
   - Handles citation generation and verification
   - Methods:
     - `find_sources(response, context)`: Find sources for statements in the response
     - `verify_response(response, context)`: Verify the response against the context

### Visualization

#### `visualization.py`

This module will implement visualization tools for knowledge graphs:

1. **`GraphVisualizer`**
   - Provides visualization of knowledge graphs
   - Methods:
     - `visualize_graph(graph, **kwargs)`: Visualize the entire graph
     - `visualize_path(graph, path, **kwargs)`: Visualize a specific path
     - `visualize_community(graph, community, **kwargs)`: Visualize a community
     - `export_to_html(graph, path)`: Export the visualization to an HTML file

2. **`VisualizationOptions`**
   - Configuration options for visualization
   - Properties:
     - `node_size`: Size of nodes
     - `node_color`: Color of nodes
     - `edge_color`: Color of edges
     - `font_size`: Size of labels
     - `layout`: Graph layout algorithm

## Data Flow

### Graph Building Flow

1. **Document Loading**: `DocumentLoader` loads documents from various sources
2. **Document Processing**: `DocumentProcessor` processes documents into chunks
3. **Graph Building**:
   - `SemanticGraphBuilder`: Builds a graph based on semantic similarity
   - `EntityGraphBuilder`: Extracts entities and relationships using LLMs
   - `HybridGraphBuilder`: Combines both approaches

### RAG Flow

1. **Query Processing**: Parse and process the user query
2. **Context Retrieval**:
   - `VectorRetriever`: Retrieves documents based on vector similarity
   - `GraphRetriever`: Retrieves documents based on graph relationships
   - `PathRetriever`: Retrieves documents based on path traversal
3. **Context Building**: Combine retrieved documents into a coherent context
4. **Response Generation**: Generate a response using an LLM
5. **Citation Generation**: Find sources for statements in the response

## Configuration

The system will be configurable through YAML files, with the following structure:

```yaml
# Example configuration
loader:
  bypass_textractor: true  # Use direct text extraction for simple formats
  min_length: 100          # Minimum chunk length
  max_length: 1000         # Maximum chunk length

embeddings:
  path: "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
  content: true            # Store content in embeddings
  hybrid: true             # Use hybrid search

graph:
  builder: "hybrid"        # Graph building strategy (semantic, entity, hybrid)
  similarity_threshold: 0.75
  llm_model: "TheBloke/Mistral-7B-OpenOrca-AWQ"
  
rag:
  retriever: "path"        # Retrieval strategy (vector, graph, path)
  generator:
    model: "TheBloke/Mistral-7B-OpenOrca-AWQ"
    template: "system\nYou are a helpful assistant...\n{text}"
  citation:
    enabled: true
    method: "semantic_similarity"
```

## CLI Interface

The CLI will be extended with new commands for graph building, traversal, and RAG:

```
# Graph building
kb graph build --type semantic --similarity 0.75 --input-dir /path/to/documents

# Entity extraction
kb graph extract --model "TheBloke/Mistral-7B-OpenOrca-AWQ" --input-dir /path/to/documents

# Graph traversal
kb graph traverse --query "Ancient Rome" --path "MATCH P=({id: 'Roman Empire'})-[*1..3]->({id: 'Battle of Hastings'}) RETURN P"

# RAG
kb rag --query "Tell me about the fall of the Roman Empire" --retriever path --path-expression "MATCH P=({id: 'Roman Empire'})-[*1..3]->({id: 'Fall of Rome'}) RETURN P"

# Visualization
kb graph visualize --output graph.html --community-detection true
```

## Implementation Roadmap

1. **Phase 1: Core Graph Building**
   - Implement `SemanticGraphBuilder`
   - Extend `kg.py` to support the new graph building approach
   - Update CLI with graph building commands

2. **Phase 2: Entity Extraction**
   - Implement `EntityGraphBuilder`
   - Add LLM integration for entity extraction
   - Update CLI with entity extraction commands

3. **Phase 3: Graph Traversal**
   - Implement `GraphTraversal` and `PathQuery`
   - Add support for Cypher-like queries
   - Update CLI with traversal commands

4. **Phase 4: RAG Pipeline**
   - Implement `RAGPipeline` and retrieval strategies
   - Add LLM integration for response generation
   - Implement citation generation
   - Update CLI with RAG commands

5. **Phase 5: Visualization**
   - Implement `GraphVisualizer`
   - Add support for different visualization options
   - Update CLI with visualization commands

## Best Practices

1. **Modularity**: Keep components loosely coupled and focused on specific responsibilities
2. **Configuration-Driven**: Make behavior configurable through YAML files
3. **Progressive Enhancement**: Allow basic functionality without requiring all components
4. **Comprehensive Testing**: Test each component individually and in integration
5. **Documentation**: Provide clear documentation for each component and its configuration options
6. **Performance Optimization**: Optimize for performance with large knowledge graphs
7. **User Experience**: Design the CLI for ease of use and clear feedback

## Conclusion

This design provides a comprehensive framework for implementing advanced knowledge graph and RAG capabilities in the `data_tools` module. By following this specification, you can create a flexible, powerful system that leverages the best practices from txtai example notebooks 55, 57, and 58.
