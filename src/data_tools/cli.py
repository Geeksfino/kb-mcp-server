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
                        recursive=True,
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

def retrieve_command(args):
    """
    Handle retrieve command.
    """
    try:
        # Create application
        print(f"Creating application with path: {args.embeddings}")
        app = Application(f"path: {args.embeddings}")

        # Perform search
        print(f"Performing search with query: {args.query}")
        results = app.search(args.query, limit=args.limit, graph=args.graph)

        # Print results
        print(f"Search results: {results}")
        print(f"Results for query: '{args.query}'")
        if args.graph:
            # Iterate over centrality nodes
            for x in list(results.centrality().keys())[:args.limit]:
                print(results.node(x))
                print()
        else:
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"  Score: {result['score']:.4f}")
                print(f"  Text: {result['text']}")
                print()

    except Exception as e:
        print(f"Error during retrieval: {e}")
        logger.error(f"Error during retrieval: {e}")

def generate_command(args):
    """
    Handle generate command.
    """
    print("Generate command not implemented yet.")

def main():
    """
    Main entry point for the Knowledge Base CLI.
    """
    parser = argparse.ArgumentParser(description="Knowledge Base CLI")
    
    # Global arguments
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    subparsers = parser.add_subparsers(title="commands", dest="command", help="Available commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build embeddings database")
    build_parser.add_argument("--input", type=str, nargs="+", help="Path to input files or directories")
    build_parser.add_argument("--extensions", type=str, help="Comma-separated list of file extensions to include")
    build_parser.add_argument("--json_input", type=str, help="Path to JSON file containing a list of documents")
    build_parser.add_argument("--config", type=str, help="Path to configuration file")
    build_parser.set_defaults(func=build_command)
    
    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve information from embeddings database")
    retrieve_parser.add_argument("embeddings", type=str, help="Path to embeddings database")
    retrieve_parser.add_argument("query", type=str, help="Search query")
    retrieve_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    retrieve_parser.add_argument("--graph", action="store_true", help="Enable graph search")
    retrieve_parser.set_defaults(func=retrieve_command)
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate answer using LLM")
    generate_parser.set_defaults(func=generate_command)
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.debug if hasattr(args, 'debug') else False)
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
