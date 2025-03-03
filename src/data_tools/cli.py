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
from data_tools.graph_traversal import GraphTraversal
from data_tools.visualization import GraphVisualizer, VisualizationOptions

# Import RAG components
from data_tools.rag import RAGPipeline, TxtaiRetriever, generate_questions_from_graph

# Import settings
from data_tools.settings import Settings
from data_tools.processor import DocumentProcessor

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

def create_application(config_path: Optional[str] = None) -> Application:
    """
    Create a txtai application with the specified configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        txtai.app.Application: Application instance
    """
    # Use provided config if available
    if config_path:
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
        
        if os.path.exists(config_path):
            logger.info(f"Loading configuration from {config_path}")
            try:
                # Create application directly from YAML file path
                app = Application(config_path)
                
                # Log configuration details
                if hasattr(app.embeddings, 'graph') and app.embeddings.graph:
                    logger.info("Graph configuration found in embeddings")
                
                # Log index path
                if hasattr(app, 'config') and 'path' in app.config:
                    logger.info(f"Index will be stored at: {app.config['path']}")
                
                return app
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
                logger.warning("Falling back to default configuration")
        else:
            logger.warning(f"Configuration file not found: {config_path}")
            logger.warning("Falling back to default configuration")
    else:
        logger.info("No configuration file specified, using default configuration")
    
    # If no config provided or loading failed, use default settings
    logger.info("Creating application with default configuration")
    
    # Get settings
    settings = Settings(config_path)
    
    # Create default configuration
    config = {
        "path": ".txtai/index",  # Default index path
        "writable": True,  # Enable index writing
        "content": True,   # Store document content
        "embeddings": {
            "path": settings.get("model_path", "sentence-transformers/all-MiniLM-L6-v2"),
            "gpu": settings.get("model_gpu", True),
            "normalize": settings.get("model_normalize", True),
            "content": True,  # Store document content
            "writable": True   # Enable index writing
        },
        "search": {
            "hybrid": settings.get("hybrid_search", False)
        }
    }
    
    return Application(config)

def build_command(args):
    """
    Handle build command.
    
    Args:
        args: Command-line arguments
    """
    # Use config from args or global args
    config_path = args.config if hasattr(args, 'config') and args.config else args.global_config
    
    if config_path:
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
            
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            logger.error("Please provide a valid path to a configuration file")
            return
        
        logger.info(f"Using configuration from {config_path}")
    else:
        logger.warning("No configuration file specified, using default settings")
    
    # Create application
    app = create_application(config_path)
    
    # Create document loader
    loader = DocumentLoader()
    
    # Process documents
    documents = []
    
    # Process JSON input if provided
    if args.json_input:
        try:
            with open(args.json_input, 'r') as f:
                json_data = json.load(f)
                
            # Check if it's a list of documents
            if isinstance(json_data, list):
                documents.extend(json_data)
                logger.info(f"Loaded {len(json_data)} documents from {args.json_input}")
            else:
                logger.error(f"Invalid JSON format in {args.json_input}. Expected a list of documents.")
        except Exception as e:
            logger.error(f"Error loading JSON from {args.json_input}: {e}")
    
    # Process file/directory inputs
    if args.input:
        # Parse extensions
        extensions = None
        if args.extensions:
            # Convert comma-separated string to set of extensions
            extensions = set(ext.strip().lower() for ext in args.extensions.split(","))
            # Add leading dot if not present
            extensions = {ext if ext.startswith('.') else f'.{ext}' for ext in extensions}
        
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
                        extensions=extensions
                    )
                    documents.extend(dir_docs)
                except Exception as e:
                    logger.error(f"Error processing directory {path}: {e}")
            
            else:
                logger.warning(f"Input path not found: {path}")
    
    # Check if we have documents to process
    if not documents:
        logger.error("No documents found to process")
        return
    
    logger.info(f"Processed {len(documents)} documents")
    
    # Use the application's add method which handles both indexing and saving
    logger.info("Indexing documents...")
    try:
        # Add documents to the index
        app.add(documents)
        
        # Build the index
        app.index()
        
        logger.info("Documents indexed successfully")
        
        # Log if graph was built
        if hasattr(app.embeddings, 'graph') and app.embeddings.graph:
            logger.info("Knowledge graph was automatically built based on YAML configuration")
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        return

def search_command(args):
    """
    Handle search command.
    
    Args:
        args: Command-line arguments
    """
    # Use embeddings path if provided, otherwise fall back to config
    embeddings_path = args.embeddings
    config_path = None
    
    if not embeddings_path:
        # Fall back to config if embeddings path not provided
        config_path = args.config if hasattr(args, 'config') and args.config else args.global_config
        
        if config_path:
            # Convert to absolute path if it's a relative path
            if not os.path.isabs(config_path):
                config_path = os.path.abspath(config_path)
                
            if not os.path.exists(config_path):
                logger.error(f"Configuration file not found: {config_path}")
                logger.error("Please provide a valid path to a configuration file")
                return
    
    # Create application
    if embeddings_path:
        # Create application directly from embeddings path
        logger.info(f"Loading embeddings from {embeddings_path}")
        app = Application(f"path: {embeddings_path}")
    else:
        # Create application from config
        app = create_application(config_path)
    
    # Get embeddings
    embeddings = app.embeddings
    
    # For graph-based searches, check if graph is available
    if args.search_type in ["graph", "path"] and (not hasattr(embeddings, 'graph') or not embeddings.graph):
        logger.error("No graph found in the embeddings")
        logger.error("Make sure your embeddings include a graph component")
        return
    
    # Create unified retriever with appropriate search type
    retriever = TxtaiRetriever(embeddings, {
        "search_type": args.search_type,
        "limit": args.limit,
        "max_hops": 2  # Default value, not exposed in CLI for basic search
    })
    
    # Perform search
    kwargs = {}
    if args.path_expression and args.search_type == "path":
        kwargs["path_expression"] = args.path_expression
    
    results = retriever.retrieve(args.query, **kwargs)
    
    # Display results
    if not results:
        print(f"No results found for query: {args.query}")
        return
    
    print(f"\nResults for query: '{args.query}'\n")
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        
        # Print score
        if "score" in result:
            print(f"  Score: {result['score']:.4f}")
        
        # Print text
        if "text" in result:
            text = result["text"]
            # Truncate long text
            if len(text) > 300:
                text = text[:300] + "..."
            print(f"  Text: {text}")
        
        # Print metadata if requested
        if args.show_metadata:
            # Show all fields except text and score which are already shown
            for key, value in result.items():
                if key not in ["text", "score"]:
                    print(f"  {key}: {value}")
        
        # Print path if available
        if "path" in result:
            print(f"  Path: {' -> '.join(result['path'])}")
        
        print()

def graph_traverse_command(args):
    """
    Handle graph-traverse command.
    
    Args:
        args: Command-line arguments
    """
    # Use config from args or global args
    config_path = args.config if hasattr(args, 'config') and args.config else args.global_config
    
    if config_path:
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
            
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            logger.error("Please provide a valid path to a configuration file")
            return
    
    # Create application
    app = create_application(config_path)
    
    # Get embeddings
    embeddings = app.embeddings
    
    # Check if graph is available
    if not hasattr(embeddings, 'graph') or not embeddings.graph:
        logger.error("No graph found in the embeddings")
        return
    
    try:
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
            if 'text' in result:
                text = result['text']
                if len(text) > 300:
                    text = text[:300] + "..."
                print(f"  Text: {text}")
            print()
    except Exception as e:
        logger.error(f"Error traversing graph: {e}")

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
    if not os.path.exists(args.input):
        logger.error(f"Graph directory not found: {args.input}")
        return
    
    # Load embeddings (which includes the graph)
    embeddings.load(args.input)
    
    # Get graph
    graph = embeddings.graph
    if not hasattr(embeddings, 'graph') or not embeddings.graph:
        logger.error("No graph found in the embeddings")
        return
    
    # Import visualization libraries
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        logger.error("Matplotlib and NetworkX are required for visualization")
        return
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Access the NetworkX backend for visualization
    graph_backend = graph.backend if hasattr(graph, 'backend') else None
    if not graph_backend:
        logger.error("Graph backend not available for visualization")
        return
    
    # Draw graph
    pos = nx.spring_layout(graph_backend)
    nx.draw(graph_backend, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='gray')
    
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
    if not os.path.exists(args.input):
        logger.error(f"Graph directory not found: {args.input}")
        return
    
    # Load embeddings (which includes the graph)
    embeddings.load(args.input)
    
    # Get graph
    graph = embeddings.graph
    if not hasattr(embeddings, 'graph') or not embeddings.graph:
        logger.error("No graph found in the embeddings")
        return
    
    # Access the NetworkX backend for visualization
    graph_backend = graph.backend if hasattr(graph, 'backend') else None
    if not graph_backend:
        logger.error("Graph backend not available for visualization")
        return
    
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
    path_graph = graph_backend.subgraph(path)
    
    # Draw graph
    pos = nx.spring_layout(path_graph)
    nx.draw(path_graph, pos, with_labels=True, node_color='lightgreen', node_size=1500, edge_color='gray')
    
    # Save figure
    plt.savefig(args.output)
    logger.info(f"Path visualization saved to {args.output}")

def rag_command(args):
    """
    Handle rag command.
    
    Args:
        args: Command-line arguments
    """
    
    # Get config path
    config_path = args.config
    
    # Create application
    app = create_application(config_path)
    
    # Create RAG pipeline
    pipeline = RAGPipeline(app.embeddings, {
        "retriever": {
            "search_type": args.search_type,
            "limit": args.limit,
            "max_hops": args.max_hops
        },
        "generator": {
            "prompt_template": args.prompt
        },
        "citation": {
            "enabled": args.citations
        }
    })
    
    # Generate response
    response = pipeline.generate(
        args.query,
        add_citations=args.citations
    )
    
    # Print response
    print("\nRAG Response:\n")
    print(response)
    
    # Print context if requested
    if args.show_context:
        print("\nRetrieved Context:\n")
        context = pipeline.retrieve(args.query)
        
        for i, doc in enumerate(context, 1):
            print(f"Document {i}:")
            print(f"  Score: {doc.get('score', 0.0):.4f}")
            
            # Print ID if available
            if "id" in doc:
                print(f"  ID: {doc['id']}")
                
            # Print title if available
            if "title" in doc:
                print(f"  Title: {doc['title']}")
                
            # Print text (truncated if too long)
            if "text" in doc:
                text = doc["text"]
                if len(text) > 300:
                    text = text[:300] + "..."
                print(f"  Text: {text}")
                
            # Print separator between documents
            print()

def graph_search_command(args):
    """
    Handle graph-search command.
    
    Args:
        args: Command-line arguments
    """
    
    # Get config path
    config_path = args.config
    
    # Create application
    app = create_application(config_path)
    
    # Get embeddings
    embeddings = app.embeddings
    
    # Check if graph is available
    if not hasattr(embeddings, 'graph') or not embeddings.graph:
        logger.error("No graph found in the embeddings")
        logger.error("Make sure your config includes a 'graph' section in the embeddings configuration")
        return
    
    try:
        # Create retriever
        retriever = TxtaiRetriever(embeddings, {
            "search_type": "graph",
            "max_hops": args.max_hops,
            "limit": args.limit
        })
        
        # Perform search
        results = retriever.retrieve(args.query)
        
        # Print results
        print(f"\nGraph search results for query: '{args.query}' (max hops: {args.max_hops})\n")
        
        if not results:
            print("No results found")
            return
            
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Score: {result.get('score', 0.0):.4f}")
            
            # Print node ID if available
            if "id" in result:
                print(f"  Node ID: {result['id']}")
                
            # Print title if available
            if "title" in result:
                print(f"  Title: {result['title']}")
                
            # Print text (truncated if too long)
            if "text" in result:
                text = result["text"]
                if len(text) > 300:
                    text = text[:300] + "..."
                print(f"  Text: {text}")
                
            # Print separator between results
            print()
            
    except Exception as e:
        logger.error(f"Error during graph search: {str(e)}")
        
        # Try direct SQL query as a fallback
        try:
            print("\nAttempting fallback search using direct SQL query...\n")
            
            # Execute a direct graph query
            sql_query = f"select id, text, title, score from txtai where graph('{args.query}', {args.max_hops}, {args.limit})"
            results = embeddings.search(sql_query)
            
            if not results:
                print("No results found with fallback method")
                return
                
            for i, result in enumerate(results, 1):
                print(f"Result {i}:")
                print(f"  Score: {result.get('score', 0.0):.4f}")
                
                # Print node ID if available
                if "id" in result:
                    print(f"  Node ID: {result['id']}")
                    
                # Print title if available
                if "title" in result:
                    print(f"  Title: {result['title']}")
                    
                # Print text (truncated if too long)
                if "text" in result:
                    text = result["text"]
                    if len(text) > 300:
                        text = text[:300] + "..."
                    print(f"  Text: {text}")
                    
                # Print separator between results
                print()
                
        except Exception as fallback_error:
            logger.error(f"Fallback search also failed: {str(fallback_error)}")
            print("\nTry checking your graph configuration and ensure the graph component is properly initialized.")

def graph_questions_command(args):
    """
    Handle graph-questions command.
    
    Args:
        args: Command-line arguments
    """
    # Use config from args or global args
    config_path = args.config if hasattr(args, 'config') and args.config else args.global_config
    
    if config_path:
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
            
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            logger.error("Please provide a valid path to a configuration file")
            return
    
    # Create application
    app = create_application(config_path)
    
    # Get embeddings
    embeddings = app.embeddings
    
    # Check if graph is available
    if not hasattr(embeddings, 'graph') or not embeddings.graph:
        logger.error("No graph found in the embeddings")
        logger.error("Make sure your config includes a 'graph' section in the embeddings configuration")
        return
    
    try:
        # Import the question generator
        from data_tools.rag import generate_questions_from_graph
        
        # Generate questions from the graph
        questions = generate_questions_from_graph(
            embeddings.graph,
            num_nodes=args.num_nodes,
            num_questions=args.num_questions
        )
        
        # Print questions
        print(f"\nGenerated questions from the knowledge graph:\n")
        
        if not questions:
            print("No questions could be generated.")
            return
            
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question}")
            
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        import traceback
        logger.debug(traceback.format_exc())

def main():
    """
    Main entry point for the Knowledge Base CLI.
    """
    parser = argparse.ArgumentParser(description="Knowledge Base CLI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build knowledge base")
    build_parser.add_argument("--config", "-c", type=str, help="Configuration file path (overrides global config)")
    build_parser.add_argument("--input", "-i", nargs="+", help="Input files or directories")
    build_parser.add_argument("--json-input", "-j", type=str, help="JSON input file")
    build_parser.add_argument("--recursive", "-r", action="store_true", help="Process directories recursively")
    build_parser.add_argument("--extensions", "-e", type=str, help="Comma-separated list of file extensions to process")
    build_parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for document processing")
    build_parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap for document processing")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search knowledge base")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--search-type", "-t", choices=["similar", "vector", "exact", "hybrid", "graph", "path"], 
                             default="hybrid", help="Type of search to perform")
    search_parser.add_argument("--limit", "-l", type=int, default=5, help="Maximum number of results")
    search_parser.add_argument("--embeddings", "-e", type=str, help="Path to embeddings directory or archive file")
    search_parser.add_argument("--show-metadata", "-m", action="store_true", help="Show document metadata")
    search_parser.add_argument("--extract-answers", "-a", action="store_true", help="Extract answers using QA")
    search_parser.add_argument("--path-expression", type=str, help="Path expression for path-based search")
    search_parser.add_argument("--config", type=str, help="Path to configuration file (deprecated, use --embeddings instead)")
    
    # Graph traverse command
    graph_traverse_parser = subparsers.add_parser("graph-traverse", help="Traverse graph")
    graph_traverse_parser.add_argument("--query", "-q", type=str, required=True, help="Query text")
    graph_traverse_parser.add_argument("--limit", "-l", type=int, default=5, help="Maximum number of results")
    graph_traverse_parser.add_argument("--config", type=str, help="Path to configuration file")
    
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
    
    # Graph search command
    graph_search_parser = subparsers.add_parser("graph-search", help="Search knowledge graph")
    graph_search_parser.add_argument("query", help="Search query")
    graph_search_parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum number of results")
    graph_search_parser.add_argument("--max-hops", "-m", type=int, default=2, help="Maximum number of hops from seed nodes")
    graph_search_parser.add_argument("--show-metadata", action="store_true", help="Show document metadata")
    graph_search_parser.add_argument("--show-neighbors", action="store_true", help="Show connected nodes")
    graph_search_parser.add_argument("--extract-facts", "-f", action="store_true", help="Extract facts from the graph")
    graph_search_parser.add_argument("--num-facts", type=int, default=5, help="Number of facts to extract")
    graph_search_parser.add_argument("--config", type=str, help="Path to configuration file")
    
    # Graph questions command
    graph_questions_parser = subparsers.add_parser("graph-questions", help="Generate questions from the graph")
    graph_questions_parser.add_argument("--num-nodes", "-n", type=int, default=10, help="Number of central nodes to include")
    graph_questions_parser.add_argument("--num-questions", "-q", type=int, default=5, help="Number of questions to generate")
    graph_questions_parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Set up logging first
    setup_logging(args.debug)
    
    # Store global config
    args.global_config = args.config
    
    # Debug log the arguments
    logger.debug(f"Command-line arguments: {args}")
    logger.debug(f"Global config: {args.global_config}")
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Execute command
        if args.command == "build":
            build_command(args)
        elif args.command == "search":
            search_command(args)
        elif args.command == "graph-traverse":
            graph_traverse_command(args)
        elif args.command == "graph-visualize":
            graph_visualize_command(args)
        elif args.command == "path-visualize":
            visualize_path_command(args)
        elif args.command == "rag":
            rag_command(args)
        elif args.command == "graph-search":
            graph_search_command(args)
        elif args.command == "graph-questions":
            graph_questions_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        if args.debug:
            logger.exception("Detailed error information:")

if __name__ == "__main__":
    main()
