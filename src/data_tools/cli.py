#!/usr/bin/env python3
"""Command-line interface for the Knowledge Base system."""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union

from txtai.app import Application
from pydantic_settings import BaseSettings

from .loader import DocumentLoader
from .processor import DocumentProcessor
from .kg import KnowledgeGraph
from .search import KnowledgeSearch

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Settings for the Knowledge Base CLI."""
    
    # Configuration path (can be embeddings path or YAML config)
    yaml_config: Optional[str] = None
    
    # Path settings
    embeddings_path: Optional[str] = None
    graph_path: Optional[str] = ".txtai/graph.pkl"
    
    # Default model settings if no config provided
    model_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_path: str = "~/.txtai/embeddings"
    
    # Model settings
    model_gpu: bool = True
    model_normalize: bool = True
    hybrid_search: bool = True
    
    class Config:
        env_prefix = "KB_"  # Look for KB_ prefixed env vars
        env_file = ".env"

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
    Create a txtai Application instance.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        txtai Application instance
    """
    # Priority 1: Use provided config path
    if config_path and os.path.exists(config_path):
        logger.info(f"Creating Application from config path: {config_path}")
        return Application(config_path)
    
    # Priority 2: Find config file in standard locations
    config_path = find_config_file()
    if config_path:
        logger.info(f"Creating Application from found config: {config_path}")
        return Application(config_path)
    
    # Priority 3: Use default settings
    settings = Settings()
    logger.info("No config found, creating Application with default settings")
    
    config = {
        "path": settings.model_path,
        "content": True,
        "writable": True,
        "embeddings": {
            "path": settings.model_path,
            "storagepath": settings.index_path,
            "gpu": settings.model_gpu,
            "normalize": settings.model_normalize,
            "hybrid": settings.hybrid_search,
            "writable": True
        }
    }
    
    return Application(config)

def build_command(args):
    """
    Handle knowledge base building command.
    
    Args:
        args: Command-line arguments
    """
    # Create txtai Application
    app = create_application(args.config)
    
    # Initialize document loader and processor
    loader = DocumentLoader(app)
    processor = DocumentProcessor(app)
    
    # Process all sources
    all_documents = []
    for source in args.sources:
        source_path = Path(source)
        
        if source_path.is_file():
            # Process a single file
            logger.info(f"Processing file: {source_path}")
            # Use direct text extraction for simple file formats
            documents = loader.process_file(source_path, bypass_textractor=True)
            logger.info(f"Extracted {len(documents)} segments from {source_path}")
            all_documents.extend(documents)
            
        elif source_path.is_dir():
            # Process a directory
            logger.info(f"Processing directory: {source_path}")
            documents = loader.process_directory(
                source_path, 
                recursive=args.recursive,
                extensions=set(args.extensions) if args.extensions else None
            )
            logger.info(f"Extracted {len(documents)} segments from directory {source_path}")
            all_documents.extend(documents)
            
        else:
            logger.warning(f"Source not found: {source_path}")
    
    # Process documents into embeddings
    if all_documents:
        logger.info(f"Processing {len(all_documents)} documents into embeddings")
        processor.process_documents(all_documents)
        
        # Save embeddings
        output_path = args.output if args.output else None
        saved_path = processor.save_embeddings(output_path)
        logger.info(f"Saved embeddings to {saved_path}")
        
        # Build knowledge graph if requested
        if args.build_graph:
            from .kg import KnowledgeGraph
            
            logger.info("Building knowledge graph")
            kg = KnowledgeGraph(app)
            graph = kg.build_graph()
            
            # Find communities if requested
            if args.find_communities:
                logger.info("Finding communities in knowledge graph")
                communities = kg.find_communities(graph)
                logger.info(f"Found {len(communities)} communities")
            
            # Save graph
            graph_path = args.graph_path if args.graph_path else os.path.join(os.path.dirname(saved_path), "graph.pkl")
            kg.save_graph(graph, graph_path)
            logger.info(f"Saved knowledge graph to {graph_path}")
    else:
        logger.warning("No documents found to process")

def search_command(args):
    """
    Handle search command.
    
    Args:
        args: Command-line arguments
    """
    # Create txtai Application
    app = create_application(args.config)
    
    # Initialize search component
    search = KnowledgeSearch(app, args.graph_path)
    
    # Perform search based on search type
    if args.search_type == "similar":
        logger.info(f"Performing similarity search for query: {args.query}")
        results = search.similarity_search(args.query, args.limit)
        
    elif args.search_type == "exact":
        logger.info(f"Performing exact match search for query: {args.query}")
        results = search.exact_search(args.query, args.limit)
        
    elif args.search_type == "hybrid":
        logger.info(f"Performing hybrid search for query: {args.query}")
        results = search.hybrid_search(args.query, args.limit)
        
    elif args.search_type == "graph":
        logger.info(f"Performing graph search for query: {args.query}")
        results = search.graph_search(args.query, args.limit, args.depth)
        
    else:
        logger.error(f"Unknown search type: {args.search_type}")
        return
    
    # Display results
    if not results:
        print("No results found")
        return
    
    print(f"\nFound {len(results)} results for query: '{args.query}'\n")
    
    for i, result in enumerate(results):
        print(f"Result {i+1} (Score: {result.get('score', 0):.4f}):")
        print(f"  ID: {result.get('id', 'N/A')}")
        
        # Print snippet or text
        snippet = result.get('snippet') or result.get('text', '')[:200]
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        print(f"  Text: {snippet}")
        
        # Print metadata if available and requested
        if args.show_metadata and 'metadata' in result:
            metadata = result['metadata']
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    pass
            
            print("  Metadata:")
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    if key != 'entities':  # Skip entities for brevity
                        print(f"    {key}: {value}")
            else:
                print(f"    {metadata}")
        
        print()
    
    # Perform extractive QA if requested
    if args.extract_answers:
        print("\nExtractive QA Results:")
        answers = search.extractive_qa(args.query, args.limit)
        
        if not answers:
            print("No answers extracted")
        else:
            for i, answer in enumerate(answers):
                print(f"Answer {i+1} (Score: {answer.get('score', 0):.4f}):")
                print(f"  {answer.get('text', 'N/A')}")
                print()

def main():
    """Main entry point for the Knowledge Base CLI."""
    parser = argparse.ArgumentParser(description="Knowledge Base CLI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build knowledge base")
    build_parser.add_argument("sources", nargs="+", help="Source files or directories to process")
    build_parser.add_argument("--recursive", "-r", action="store_true", help="Process directories recursively")
    build_parser.add_argument("--extensions", "-e", nargs="+", help="File extensions to process")
    build_parser.add_argument("--output", "-o", type=str, help="Output path for embeddings")
    build_parser.add_argument("--build-graph", action="store_true", help="Build knowledge graph")
    build_parser.add_argument("--graph-path", type=str, help="Path to save knowledge graph")
    build_parser.add_argument("--find-communities", action="store_true", help="Find communities in knowledge graph")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search knowledge base")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--search-type", "-t", choices=["similar", "exact", "hybrid", "graph"], 
                              default="hybrid", help="Type of search to perform")
    search_parser.add_argument("--limit", "-l", type=int, default=5, help="Maximum number of results")
    search_parser.add_argument("--depth", "-d", type=int, default=2, help="Maximum depth for graph traversal")
    search_parser.add_argument("--graph-path", type=str, help="Path to knowledge graph (for graph search)")
    search_parser.add_argument("--show-metadata", "-m", action="store_true", help="Show document metadata")
    search_parser.add_argument("--extract-answers", "-a", action="store_true", help="Extract answers using QA")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.debug)
    
    # Execute command
    if args.command == "build":
        build_command(args)
    elif args.command == "search":
        search_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
