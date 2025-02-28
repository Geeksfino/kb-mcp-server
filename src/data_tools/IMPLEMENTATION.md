# Knowledge Graph and RAG Implementation

This document provides an overview of the implementation of the Knowledge Graph and Retrieval Augmented Generation (RAG) functionality in the system.

## Architecture Overview

The system is designed with a modular architecture that separates concerns and allows for extensibility. The main components are:

1. **Graph Building**: Responsible for creating knowledge graphs from documents
2. **Graph Traversal**: Provides path-based traversal and querying of knowledge graphs
3. **RAG Pipeline**: Coordinates the retrieval of context and generation of responses
4. **Visualization**: Provides visualization tools for knowledge graphs

## Graph Building

The graph building module (`graph_builder.py`) provides three types of graph builders:

### SemanticGraphBuilder

Builds graphs based on semantic similarity between documents. The process is:

1. Compute embeddings for all documents
2. For each document, find the most similar documents based on cosine similarity
3. Create edges between documents that exceed a similarity threshold
4. Limit the number of connections per document to avoid dense graphs

### EntityGraphBuilder

Extracts entities and relationships using Large Language Models (LLMs). The process is:

1. For each document, use an LLM to extract entities and their relationships
2. Create nodes for each entity
3. Create edges between entities based on the extracted relationships
4. Assign attributes to nodes and edges based on the extracted information

### HybridGraphBuilder

Combines both semantic and entity approaches to create a more comprehensive graph. The process is:

1. Build a semantic graph using the SemanticGraphBuilder
2. Build an entity graph using the EntityGraphBuilder
3. Merge the two graphs, combining nodes and edges
4. Resolve conflicts and duplicates

## Graph Traversal

The graph traversal module (`graph_traversal.py`) provides path-based traversal and querying of knowledge graphs:

### GraphTraversal

Provides methods for traversing the graph and finding paths between nodes. Key features:

1. Finding paths between nodes with customizable hop limits
2. Ranking paths based on relevance to a query
3. Extracting information from nodes along a path

### PathQuery

Implements a Cypher-like query language for expressing complex path patterns. Examples:

- `(a)-[r]->(b)`: Simple path from node a to node b
- `(a)-[r]->(b)-[s]->(c)`: Path from node a to node c through node b
- `(a)-[r*1..3]->(b)`: Path from node a to node b with 1 to 3 hops

## RAG Pipeline

The RAG pipeline module (`rag.py`) coordinates the retrieval of context and generation of responses:

### Retrievers

Different retrieval strategies for finding relevant context:

1. **VectorRetriever**: Retrieves context based on vector similarity
2. **GraphRetriever**: Retrieves context based on graph relationships
3. **PathRetriever**: Retrieves context based on path traversal

### Generator

Responsible for generating responses using an LLM and the retrieved context. Features:

1. Prompt engineering for effective context utilization
2. Support for different LLM backends
3. Customizable generation parameters

### Citation

Manages citation generation and verification. Features:

1. Automatic citation of sources used in generation
2. Verification of generated content against source material
3. Coverage metrics for response verification

### RAGPipeline

Coordinates the retrieval and generation process. Features:

1. Flexible configuration of retrievers and generators
2. Support for different retrieval strategies
3. Citation generation and verification

## Visualization

The visualization module (`visualization.py`) provides tools for visualizing knowledge graphs:

### GraphVisualizer

Provides methods for visualizing graphs, paths, and communities. Features:

1. Static visualization with multiple layout algorithms
2. Interactive HTML-based visualizations
3. Path and community visualization

### VisualizationOptions

Configuration options for visualization. Features:

1. Customizable node and edge appearance
2. Different layout algorithms
3. Figure and label options

## CLI Interface

The CLI interface (`cli.py`) provides commands for interacting with the knowledge graph and RAG functionality:

1. **graph-build**: Build a knowledge graph from documents
2. **graph-traverse**: Traverse a knowledge graph with a query
3. **graph-visualize**: Visualize a knowledge graph
4. **rag**: Run the RAG pipeline with a query

## Configuration

The system uses a configuration-driven approach, with settings available through:

1. Command-line arguments
2. Environment variables
3. Configuration files

## Integration with txtai

The system integrates with txtai for:

1. Embeddings and similarity search
2. Document processing
3. LLM access

## Dependencies

Key dependencies include:

1. **networkx**: For graph management and traversal
2. **txtai**: For embeddings, LLM access, and document processing
3. **matplotlib** and **pyvis**: For visualization
