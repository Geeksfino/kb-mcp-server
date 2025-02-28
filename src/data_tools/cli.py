#!/usr/bin/env python3
"""
CLI for Knowledge Base operations.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Import txtai
from txtai.app import Application
from txtai.pipeline import Extractor

# Import document loader
from data_tools.loader import DocumentLoader

# Import graph components
from data_tools.graph_builder import SemanticGraphBuilder, EntityGraphBuilder, HybridGraphBuilder
from data_tools.graph_traversal import GraphTraversal
from data_tools.visualization import GraphVisualizer, VisualizationOptions

# Import RAG components
from data_tools.rag import RAGPipeline
from data_tools.rag import VectorRetriever, GraphRetriever, PathRetriever

# Import settings
from data_tools.settings import Settings

# Configure logging
logger = logging.getLogger(__name__)

def setup_logging(debug: bool = False):
    """Set up logging configuration.
    
    Args:
        debug: Whether to enable debug logging
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

def find_config_file() -> Optional[str]:
    """
    Find a configuration file in standard locations.
    
    Returns:
        Path to the configuration file if found, None otherwise.
    """
    # Check environment variable first
    if os.environ.get("KB_CONFIG"):
        config_path = os.environ.get("KB_CONFIG")
        if os.path.exists(config_path):
            return config_path
    
    # Check standard locations
    search_paths = [
        "./config.yaml",
        "./config.yml",
        Path.home() / ".config" / "knowledge-base" / "config.yaml",
        Path.home() / ".config" / "knowledge-base" / "config.yml",
        Path.home() / ".knowledge-base" / "config.yaml",
        Path.home() / ".knowledge-base" / "config.yml",
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return str(path)
    
    return None

def create_application(config_path=None):
    """
    Create a txtai Application with the given configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        txtai.app.Application: Application instance
    """
    # Get settings
    settings = Settings()
    
    # Use provided config if available
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        return Application(config_path)
    
    # Use settings.yaml_config if available
    if settings.yaml_config and os.path.exists(settings.yaml_config):
        logger.info(f"Loading configuration from {settings.yaml_config}")
        return Application(settings.yaml_config)
    
    # Create default configuration
    logger.info("Creating default configuration")
    config = {
        "path": settings.index_path,
        "embeddings": {
            "path": settings.model_path,
            "gpu": settings.model_gpu,
            "normalize": settings.model_normalize
        },
        "search": {
            "hybrid": settings.hybrid_search
        }
    }
    
    return Application(config)

def build_command(args):
    """
    Handle build command.
    
    Args:
        args: Command-line arguments
    """
    # Create application
    app = create_application(args.config)
    
    # Create document loader
    loader = DocumentLoader(app)
    
    # Process input files
    documents = []
    
    # Process JSON input if provided
    if args.json_input:
        if not os.path.exists(args.json_input):
            logger.error(f"JSON input file not found: {args.json_input}")
            return
        
        import json
        with open(args.json_input, "r") as f:
            json_data = json.load(f)
        
        if isinstance(json_data, list):
            documents.extend(json_data)
        else:
            documents.append(json_data)
        
        logger.info(f"Loaded {len(documents)} documents from JSON")
    
    # Process file/directory inputs
    if args.input:
        # Parse extensions
        extensions = None
        if args.extensions:
            extensions = args.extensions.split(",")
        
        # Parse exclude patterns
        exclude_patterns = None
        if args.exclude:
            exclude_patterns = args.exclude.split(",")
        
        for input_path in args.input:
            path = Path(input_path)
            
            if path.is_file():
                logger.info(f"Processing file: {path}")
                try:
                    file_docs = loader.process_file(str(path))
                    documents.extend(file_docs)
                except Exception as e:
                    logger.error(f"Error processing file {path}: {e}")
            
            elif path.is_dir():
                logger.info(f"Processing directory: {path}")
                try:
                    dir_docs = loader.process_directory(
                        str(path),
                        recursive=args.recursive,
                        extensions=extensions,
                        exclude_patterns=exclude_patterns
                    )
                    documents.extend(dir_docs)
                except Exception as e:
                    logger.error(f"Error processing directory {path}: {e}")
            
            else:
                logger.warning(f"Input path not found: {path}")
    
    if not documents:
        logger.error("No documents found to process")
        return
    
    logger.info(f"Processed {len(documents)} documents")
    
    # Index documents
    logger.info("Indexing documents...")
    app.add(documents)
    app.index()
    logger.info("Documents indexed successfully")
    
    # Build graph if requested
    if args.build_graph:
        logger.info("Building knowledge graph...")
        
        # Create graph builder
        builder_type = args.graph_builder if args.graph_builder else Settings().graph_builder
        
        # Select graph builder based on type
        if builder_type == "semantic":
            builder = SemanticGraphBuilder(app.embeddings)
        elif builder_type == "entity":
            builder = EntityGraphBuilder(app.embeddings)
        elif builder_type == "hybrid":
            builder = HybridGraphBuilder(app.embeddings)
        else:
            logger.warning(f"Unknown graph builder type: {builder_type}, using semantic")
            builder = SemanticGraphBuilder(app.embeddings)
        
        # Configure builder
        builder.similarity_threshold = args.similarity_threshold if args.similarity_threshold else Settings().similarity_threshold
        builder.max_connections = args.max_connections if args.max_connections else Settings().max_connections
        
        # Build graph
        graph = builder.build(documents)
        
        # Save graph
        graph_path = args.graph_path if args.graph_path else Settings().graph_path
        builder.save(graph_path)
        
        logger.info(f"Graph built with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        logger.info(f"Graph saved to {graph_path}")
    
    logger.info("Build completed successfully")

def search_command(args):
    """
    Handle search command.
    
    Args:
        args: Command-line arguments
    """
    # Create application
    app = create_application(args.config)
    
    # Get embeddings
    embeddings = app.embeddings
    
    # Create retriever based on search type
    kwargs = {}
    
    if args.search_type == "vector" or args.search_type == "similar":
        # Vector-based search
        retriever = VectorRetriever(embeddings)
        
    elif args.search_type == "graph":
        # Graph-based search
        if not args.graph_path:
            logger.error("Graph path is required for graph search")
            return
        
        # Load embeddings with graph
        if not os.path.exists(os.path.dirname(args.graph_path)):
            logger.error(f"Graph directory not found: {os.path.dirname(args.graph_path)}")
            return
        
        # Load embeddings (which includes the graph)
        embeddings.load(os.path.dirname(args.graph_path))
        
        # Create graph retriever
        retriever = GraphRetriever(embeddings)
        
    elif args.search_type == "path":
        # Path-based search
        if not args.graph_path:
            logger.error("Graph path is required for path search")
            return
        
        # Load embeddings with graph
        if not os.path.exists(os.path.dirname(args.graph_path)):
            logger.error(f"Graph directory not found: {os.path.dirname(args.graph_path)}")
            return
        
        # Load embeddings (which includes the graph)
        embeddings.load(os.path.dirname(args.graph_path))
        
        # Create path retriever
        retriever = PathRetriever(embeddings)
        
    elif args.search_type == "hybrid":
        # Hybrid search (vector + graph)
        if not args.graph_path:
            logger.warning("Graph path not provided, falling back to vector search")
            retriever = VectorRetriever(embeddings)
        else:
            # Load embeddings with graph
            if not os.path.exists(os.path.dirname(args.graph_path)):
                logger.warning(f"Graph directory not found: {os.path.dirname(args.graph_path)}, falling back to vector search")
                retriever = VectorRetriever(embeddings)
            else:
                # Load embeddings (which includes the graph)
                embeddings.load(os.path.dirname(args.graph_path))
                
                # Create vector retriever as fallback
                retriever = VectorRetriever(embeddings)
                
    elif args.search_type == "exact":
        # Exact search
        logger.warning("Exact search not implemented, falling back to vector search")
        retriever = VectorRetriever(embeddings)
    
    else:
        logger.error(f"Unknown search type: {args.search_type}")
        return
    
    # Perform search
    kwargs = {}
    if args.path_expression and args.search_type == "path":
        kwargs["path_expression"] = args.path_expression
    
    results = retriever.retrieve(args.query, limit=args.limit, **kwargs)
    
    # Display results
    if not results:
        print(f"No results found for query: {args.query}")
        return
    
    print(f"\nResults for query: '{args.query}'\n")
    
    for i, result in enumerate(results):
        print(f"Result {i+1} (Score: {result.get('score', 0):.4f}):")
        
        # Print ID
        print(f"  ID: {result.get('id', 'N/A')}")
        
        # Print text
        text = result.get('text', '')
        if text and len(text) > 200:
            text = text[:200] + "..."
        print(f"  Text: {text}")
        
        # Print metadata if requested
        if args.show_metadata and 'metadata' in result:
            print("  Metadata:")
            for key, value in result['metadata'].items():
                print(f"    {key}: {value}")
        
        print()
    
    # Extract answers if requested
    if args.extract_answers:
        from txtai.pipeline import Extractor
        
        # Create extractor
        extractor = Extractor()
        
        # Extract answer
        answer = extractor(args.query, [(r["id"], r["text"]) for r in results])
        
        print("\nExtracted Answer:")
        print(answer)

def graph_build_command(args):
    """
    Handle graph-build command.
    
    Args:
        args: Command-line arguments
    """
    # Create application
    app = create_application(args.config)
    
    # Process input
    documents = []
    if os.path.isdir(args.input):
        logger.info(f"Processing directory: {args.input}")
        loader = DocumentLoader(app)
        documents = loader.process_directory(args.input)
    elif os.path.isfile(args.input):
        logger.info(f"Processing file: {args.input}")
        loader = DocumentLoader(app)
        documents = loader.process_file(args.input)
    else:
        logger.error(f"Input path not found: {args.input}")
        return
    
    logger.info(f"Building graph from {len(documents)} documents")
    
    # Create graph builder
    if args.builder == "semantic":
        builder = SemanticGraphBuilder(app.embeddings)
    elif args.builder == "entity":
        builder = EntityGraphBuilder(app.embeddings)
    elif args.builder == "hybrid":
        builder = HybridGraphBuilder(app.embeddings)
    else:
        logger.error(f"Unknown graph builder: {args.builder}")
        return
    
    # Build graph
    graph = builder.build(documents)
    
    # Save graph
    output_path = args.output if args.output else os.path.join(app.config.get("path", ".txtai"), "graph.pkl")
    builder.save(output_path)
    
    # Log graph info
    logger.info(f"Graph built with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    logger.info(f"Graph saved to {output_path}")

def graph_traverse_command(args):
    """
    Handle graph-traverse command.
    
    Args:
        args: Command-line arguments
    """
    # Create application
    app = create_application(args.config)
    
    # Get embeddings
    embeddings = app.embeddings
    
    # Load embeddings with graph
    if not os.path.exists(os.path.dirname(args.graph)):
        logger.error(f"Graph directory not found: {os.path.dirname(args.graph)}")
        return
    
    # Load embeddings (which includes the graph)
    embeddings.load(os.path.dirname(args.graph))
    
    # Create traversal
    traversal = GraphTraversal(embeddings)
    
    # Traverse graph
    results = traversal.traverse(args.query, limit=args.limit)
    
    # Print results
    print(f"\nTraversal results for query: '{args.query}'\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Path: {' -> '.join(result['path'])}")
        print(f"  Score: {result['score']:.4f}")
        print()

def graph_visualize_command(args):
    """
    Handle graph-visualize command.
    
    Args:
        args: Command-line arguments
    """
    # Create application
    app = create_application(args.config)
    
    # Get embeddings
    embeddings = app.embeddings
    
    # Load embeddings with graph
    if not os.path.exists(os.path.dirname(args.input)):
        logger.error(f"Graph directory not found: {os.path.dirname(args.input)}")
        return
    
    # Load embeddings (which includes the graph)
    embeddings.load(os.path.dirname(args.input))
    
    # Get graph
    graph = embeddings.graph
    
    # Import visualization libraries
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        logger.error("Matplotlib and NetworkX are required for visualization")
        return
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Draw graph
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray')
    
    # Save figure
    plt.savefig(args.output)
    logger.info(f"Graph visualization saved to {args.output}")

def visualize_path_command(args):
    """
    Handle path-visualize command.
    
    Args:
        args: Command-line arguments
    """
    # Create application
    app = create_application(args.config)
    
    # Get embeddings
    embeddings = app.embeddings
    
    # Load embeddings with graph
    if not os.path.exists(os.path.dirname(args.input)):
        logger.error(f"Graph directory not found: {os.path.dirname(args.input)}")
        return
    
    # Load embeddings (which includes the graph)
    embeddings.load(os.path.dirname(args.input))
    
    # Get graph
    graph = embeddings.graph
    
    # Parse path
    path = args.path.split(",")
    
    # Import visualization libraries
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        logger.error("Matplotlib and NetworkX are required for visualization")
        return
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create subgraph with path nodes
    path_graph = graph.subgraph(path)
    
    # Draw graph
    pos = nx.spring_layout(path_graph)
    nx.draw(path_graph, pos, with_labels=True, node_color='lightgreen', node_size=1500, edge_color='gray')
    
    # Save figure
    plt.savefig(args.output)
    logger.info(f"Path visualization saved to {args.output}")

def rag_command(args):
    """
    Handle RAG command.
    
    Args:
        args: Command-line arguments
    """
    # Create application
    app = create_application(args.config)
    
    # Create RAG pipeline
    embeddings = app.embeddings
    
    # Load graph if needed
    if args.retriever in ["graph", "path"]:
        if not args.graph:
            logger.error(f"Graph path is required for {args.retriever} retrieval")
            return
        
        # Load embeddings with graph
        if not os.path.exists(os.path.dirname(args.graph)):
            logger.error(f"Graph directory not found: {os.path.dirname(args.graph)}")
            return
        
        # Load embeddings (which includes the graph)
        embeddings.load(os.path.dirname(args.graph))
    
    # Create RAG config
    rag_config = {
        "generator": {
            "model": args.model if args.model else "TheBloke/Mistral-7B-OpenOrca-AWQ",
            "max_tokens": args.max_tokens
        },
        "citation": {
            "enabled": args.citations
        }
    }
    
    # Create pipeline
    pipeline = RAGPipeline(embeddings, rag_config)
    
    # Generate response
    kwargs = {}
    if args.path_expression:
        kwargs["path_expression"] = args.path_expression
    
    # If retrieval only, just show the retrieved documents
    if args.retrieval_only or args.show_context:
        # Get retriever
        retriever = None
        if args.retriever == "vector":
            retriever = VectorRetriever(embeddings)
        elif args.retriever == "graph":
            retriever = GraphRetriever(embeddings)
        elif args.retriever == "path":
            retriever = PathRetriever(embeddings)
        else:
            retriever = VectorRetriever(embeddings)
        
        # Retrieve documents
        results = retriever.retrieve(args.query, limit=5, **kwargs)
        
        print(f"\nRetrieved {len(results)} documents for query: '{args.query}'\n")
        
        for i, result in enumerate(results):
            print(f"Document {i+1} (Score: {result.get('score', 0):.4f}):")
            print(f"  ID: {result.get('id', 'N/A')}")
            
            # Print text
            text = result.get('text', '')
            if text and len(text) > 200:
                text = text[:200] + "..."
            print(f"  Text: {text}")
            print()
        
        # If we're only showing context, return
        if args.retrieval_only:
            return
    
    # Generate response with citations if requested
    if args.citations:
        result = pipeline.generate_with_citations(args.query, **kwargs)
        
        # Print response
        print("\nResponse:")
        print(result["response"])
        
        # Print citations
        print("\nCitations:")
        for statement, sources in result["citations"].items():
            print(f"\nStatement: {statement}")
            for source in sources:
                text = source.get("text", "")
                if text and len(text) > 100:
                    text = text[:100] + "..."
                print(f"  - {text}")
        
        # Print verification
        print(f"\nVerification: {result['verification']['coverage']*100:.1f}% coverage")
    else:
        response = pipeline.generate(args.query, **kwargs)
        print("\nResponse:")
        print(response)

def main():
    """Main entry point for the Knowledge Base CLI."""
    parser = argparse.ArgumentParser(description="Knowledge Base CLI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build knowledge base")
    build_parser.add_argument("--input", "-i", nargs="+", help="Input files or directories to process")
    build_parser.add_argument("--json-input", type=str, help="Path to JSON file containing documents")
    build_parser.add_argument("--recursive", "-r", action="store_true", help="Process directories recursively")
    build_parser.add_argument("--extensions", "-e", type=str, help="Comma-separated list of file extensions to process")
    build_parser.add_argument("--exclude", "-x", type=str, help="Comma-separated list of patterns to exclude")
    build_parser.add_argument("--bypass-textractor", action="store_true", help="Bypass textractor for simple file formats")
    build_parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for document processing")
    build_parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for document processing")
    build_parser.add_argument("--build-graph", action="store_true", help="Build knowledge graph")
    build_parser.add_argument("--graph-builder", type=str, choices=["semantic", "entity", "hybrid"], 
                             default="semantic", help="Type of graph builder to use")
    build_parser.add_argument("--graph-path", type=str, default=".txtai/graph.pkl", help="Path to save knowledge graph")
    build_parser.add_argument("--similarity-threshold", type=float, default=0.75, 
                             help="Similarity threshold for graph building")
    build_parser.add_argument("--max-connections", type=int, default=5, 
                             help="Maximum connections per node for graph building")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search knowledge base")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--search-type", "-t", choices=["similar", "vector", "exact", "hybrid", "graph", "path"], 
                             default="hybrid", help="Type of search to perform")
    search_parser.add_argument("--limit", "-l", type=int, default=5, help="Maximum number of results")
    search_parser.add_argument("--graph-path", type=str, default=".txtai/graph.pkl", help="Path to knowledge graph")
    search_parser.add_argument("--show-metadata", "-m", action="store_true", help="Show document metadata")
    search_parser.add_argument("--extract-answers", "-a", action="store_true", help="Extract answers using QA")
    search_parser.add_argument("--path-expression", type=str, help="Path expression for path-based search")
    
    # Graph build command
    graph_build_parser = subparsers.add_parser("graph-build", help="Build graph")
    graph_build_parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    graph_build_parser.add_argument("--output", "-o", type=str, default=".txtai/graph.pkl", help="Output graph file")
    graph_build_parser.add_argument("--builder", "-b", choices=["semantic", "entity", "hybrid"], 
                                  default="semantic", help="Type of graph builder")
    graph_build_parser.add_argument("--similarity-threshold", "-s", type=float, default=0.75, 
                                  help="Similarity threshold for connections")
    graph_build_parser.add_argument("--max-connections", "-m", type=int, default=5, 
                                  help="Maximum connections per node")
    
    # Graph traverse command
    graph_traverse_parser = subparsers.add_parser("graph-traverse", help="Traverse graph")
    graph_traverse_parser.add_argument("--graph", "-g", type=str, required=True, help="Input graph directory")
    graph_traverse_parser.add_argument("--query", "-q", type=str, required=True, help="Query text")
    graph_traverse_parser.add_argument("--limit", "-l", type=int, default=5, help="Maximum number of results")
    
    # Graph visualize command
    graph_visualize_parser = subparsers.add_parser("graph-visualize", help="Visualize graph")
    graph_visualize_parser.add_argument("--input", "-i", type=str, required=True, help="Input graph directory")
    graph_visualize_parser.add_argument("--output", "-o", type=str, default="graph.png", help="Output file")
    
    # Path visualize command
    path_visualize_parser = subparsers.add_parser("path-visualize", help="Visualize path")
    path_visualize_parser.add_argument("--input", "-i", type=str, required=True, help="Input graph directory")
    path_visualize_parser.add_argument("--path", "-p", type=str, required=True, help="Path as comma-separated node IDs")
    path_visualize_parser.add_argument("--output", "-o", type=str, default="path.png", help="Output file")
    
    # RAG command
    rag_parser = subparsers.add_parser("rag", help="Run RAG pipeline")
    rag_parser.add_argument("query", help="Query for RAG")
    rag_parser.add_argument("--retriever", "-r", choices=["vector", "graph", "path"], default="vector", 
                          help="Retrieval method")
    rag_parser.add_argument("--graph", "-g", type=str, help="Path to graph file (required for graph/path retrieval)")
    rag_parser.add_argument("--path-expression", "-p", type=str, help="Path expression for path retrieval")
    rag_parser.add_argument("--model", "-m", type=str, help="Generator model name")
    rag_parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens for generation")
    rag_parser.add_argument("--citations", "-c", action="store_true", help="Generate citations")
    rag_parser.add_argument("--retrieval-only", action="store_true", 
                          help="Only perform retrieval, don't generate text")
    rag_parser.add_argument("--show-context", action="store_true", help="Show retrieved context")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.debug)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Execute command
        if args.command == "build":
            build_command(args)
        elif args.command == "search":
            search_command(args)
        elif args.command == "graph-build":
            graph_build_command(args)
        elif args.command == "graph-traverse":
            graph_traverse_command(args)
        elif args.command == "graph-visualize":
            graph_visualize_command(args)
        elif args.command == "path-visualize":
            visualize_path_command(args)
        elif args.command == "rag":
            rag_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
    except Exception as e:
        logger.error(f"Error executing command: {e}", exc_info=args.debug)
        sys.exit(1)

if __name__ == "__main__":
    main()
