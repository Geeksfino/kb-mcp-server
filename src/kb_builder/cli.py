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

# Import settings
from .settings import Settings

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
        
    The build command processes input documents and builds a searchable index.
    When used with --update, it will add to or update an existing index instead of rebuilding it.
    It can also export the built index to a compressed tar.gz file for portability.
    """
    # Check if required arguments are provided
    if not ((hasattr(args, 'input') and args.input) or 
            (hasattr(args, 'json_input') and args.json_input)):
        logger.error("Error: No input sources provided.")
        logger.error("Please provide at least one input source using --input or --json_input")
        print("\nBuild command usage:")
        print("  python -m kb_builder build --input PATH [PATH ...] [--config CONFIG] [--export EXPORT_PATH] [--update]")
        print("  python -m kb_builder build --json_input JSON_FILE [--config CONFIG] [--export EXPORT_PATH] [--update]")
        print("\nOptions:")
        print("  --input PATH       Path to input files or directories")
        print("  --json_input PATH  Path to JSON file containing a list of documents")
        print("  --extensions EXT   Comma-separated list of file extensions to include")
        print("  --config PATH      Path to configuration file")
        print("  --export PATH      Export the built index to a compressed tar.gz file")
        print("  --update           Update existing index instead of rebuilding it")
        return
        
    # Use config from args or try to find a default config
    config_path = args.config if hasattr(args, 'config') and args.config else None
    
    # If no config provided via args, try to find one
    if not config_path:
        config_path = find_config_file()
    
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
    
    # Verify textractor pipeline exists
    if "textractor" not in app.pipelines:
        logger.error("No textractor pipeline configured in YAML. Please add a 'textractor' section to your configuration.")
        logger.error("Example: textractor:\n  paragraphs: true\n  minlength: 100")
        return
    
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
                    # Extract text using textractor pipeline
                    segments = app.pipelines["textractor"](str(path))
                    
                    # Create documents with metadata
                    for i, text in enumerate(segments):
                        doc_id = f"{path.stem}_{i}"
                        documents.append({
                            "id": doc_id,
                            "text": text,
                            "metadata": {
                                "source": str(path),
                                "index": i,
                                "total": len(segments)
                            }
                        })
                    logger.info(f"Extracted {len(segments)} segments from {path}")
                except Exception as e:
                    logger.error(f"Error processing file {path}: {e}")
            
            elif path.is_dir():
                logger.info(f"Processing directory: {path}")
                try:
                    # Find all files in directory
                    files = []
                    if extensions:
                        for ext in extensions:
                            files.extend(path.glob(f"**/*{ext}"))
                    else:
                        files = list(path.glob("**/*"))
                    
                    # Filter out directories
                    files = [f for f in files if f.is_file()]
                    
                    logger.info(f"Found {len(files)} files in directory {path}")
                    
                    # Process each file
                    for file_path in files:
                        try:
                            # Extract text using textractor pipeline
                            segments = app.pipelines["textractor"](str(file_path))
                            
                            # Create documents with metadata
                            for i, text in enumerate(segments):
                                doc_id = f"{file_path.stem}_{i}"
                                documents.append({
                                    "id": doc_id,
                                    "text": text,
                                    "metadata": {
                                        "source": str(file_path),
                                        "index": i,
                                        "total": len(segments)
                                    }
                                })
                        except Exception as e:
                            logger.error(f"Error processing file {file_path}: {e}")
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
        # Build or update the index
        if hasattr(args, 'update') and args.update:
            # Use upsert to update the existing index
            # The database.stream() method doesn't exist, directly use the documents list
            # Convert metadata to JSON string if it's a dictionary to avoid SQLite binding errors
            documents_for_upsert = []
            for i, doc in enumerate(documents):
                doc_id = doc.get("id", i)
                text = doc.get("text", "")
                metadata = doc.get("metadata")
                # Convert metadata to JSON string if it's a dictionary
                if isinstance(metadata, dict):
                    metadata = json.dumps(metadata)
                documents_for_upsert.append((doc_id, text, metadata))
            
            # Make sure the directory exists before upserting
            if app.config and "path" in app.config:
                os.makedirs(app.config["path"], exist_ok=True)
                
            # Add documents to the index
            app.embeddings.upsert(documents_for_upsert)
            logger.info("Documents added to existing index successfully")
        else:
            # Make sure the directory exists before indexing
            if app.config and "path" in app.config:
                os.makedirs(app.config["path"], exist_ok=True)
                
            # Add documents to the buffer
            app.add(documents)
            
            # Build the index
            app.index()
            logger.info("Documents indexed successfully")
        
        # Log if graph was built
        if hasattr(app.embeddings, 'graph') and app.embeddings.graph:
            logger.info("Knowledge graph was automatically built based on YAML configuration")
            
        # Explicitly save the index to the configured path
        if app.config and "path" in app.config:
            save_path = app.config["path"]
            logger.info(f"Saving index to configured path: {save_path}")
            os.makedirs(save_path, exist_ok=True)
            app.embeddings.save(save_path)
            logger.info(f"Index successfully saved to {save_path}")
            
        # Export the index to a tar.gz file if requested
        if hasattr(args, 'export') and args.export:
            export_path = args.export
            # Add .tar.gz extension if not present
            if not export_path.endswith('.tar.gz'):
                export_path = f"{export_path}.tar.gz"
                
            logger.info(f"Exporting index to {export_path}")
            try:
                # Save the embeddings to a tar.gz file
                app.embeddings.save(export_path)
                logger.info(f"Index successfully exported to {export_path}")
                
                # Show how to load this index
                logger.info(f"To load this index, use: app = Application(\"path: {export_path}\")")
            except Exception as e:
                logger.error(f"Error exporting index to {export_path}: {e}")
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
        
        # Extract key terms from the query to use for relevance boosting
        query_terms = set(args.query.lower().split())
        # Remove common stop words
        stop_words = {"what", "are", "is", "the", "for", "and", "or", "to", "in", "of", "a", "an"}
        query_terms = query_terms - stop_words
        
        # Perform the search
        results = app.search(args.query, limit=max(10, args.limit * 2), graph=args.graph)  # Get more results initially for filtering

        # Apply generic result enhancement
        if args.graph:
            # For graph results, enhance using centrality and query relevance
            if hasattr(results, 'centrality') and callable(results.centrality):
                # Get all nodes with their centrality scores
                nodes_with_scores = []
                for node_id in results.centrality().keys():
                    node = results.node(node_id)
                    if node and "text" in node:
                        # Base score from centrality
                        score = results.centrality()[node_id]
                        
                        # Boost score based on query term presence
                        text = node["text"].lower()
                        term_matches = sum(1 for term in query_terms if term in text)
                        if term_matches > 0:
                            # Boost proportional to the number of matching terms
                            score *= (1 + (0.2 * term_matches))
                            
                        # Add to candidates if score meets minimum threshold
                        if score >= args.min_similarity:
                            nodes_with_scores.append((node_id, score, node["text"]))
                
                # Sort by enhanced score and limit
                nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
                nodes_with_scores = nodes_with_scores[:args.limit]
                
                # Convert to the format expected by format_graph_results
                graph_results = [{"text": text, "score": score} for _, score, text in nodes_with_scores]
            else:
                # Fallback if centrality not available
                graph_results = []
                for x in list(results)[:args.limit]:
                    if "text" in x:
                        score = x.get("score", 0.5)
                        if score >= args.min_similarity:
                            graph_results.append({"text": x["text"], "score": score})
        else:
            # For regular search results, enhance based on query relevance
            enhanced_results = []
            for result in results:
                if "text" in result and "score" in result:
                    # Base score from search
                    score = result["score"]
                    
                    # Boost score based on query term presence
                    text = result["text"].lower()
                    term_matches = sum(1 for term in query_terms if term in text)
                    if term_matches > 0:
                        # Boost proportional to the number of matching terms
                        score *= (1 + (0.1 * term_matches))
                    
                    # Add to candidates if score meets minimum threshold
                    if score >= args.min_similarity:
                        enhanced_results.append({"text": result["text"], "score": score})
            
            # Sort by enhanced score and limit
            enhanced_results.sort(key=lambda x: x["score"], reverse=True)
            results = enhanced_results[:args.limit]

        # Print results
        if args.graph:
            # Format and print results
            if graph_results:
                formatted_results = format_graph_results(app.embeddings, graph_results, args.query)
                print(formatted_results)
            else:
                print(f"Q:{args.query}")
                print("No results found.\n")
        else:
            print(f"Results for query: '{args.query}'")
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"  Score: {result['score']:.4f}")
                print(f"  Text: {result['text']}")
                print()

    except Exception as e:
        print(f"Error during retrieval: {e}")
        logger.error(f"Error during retrieval: {e}")



def format_graph_results(embeddings, results, query=None):
    """
    Format graph search results to match graph.ipynb output format.
    """
    output = []
    
    if query:
        output.append(f"Q:{query}")
    
    # Process each result
    for result in results:
        try:
            # Get the text and metadata
            if isinstance(result, dict) and "text" in result:
                text = result["text"]
                # Generate a node id from the content
                words = text.split()[:5]  # Take first 5 words
                node_id = "_".join(w.lower() for w in words if w.isalnum())
            else:
                node = embeddings.graph.node(result)
                if not node:
                    continue
                text = node.get("text", "")
                node_id = result
            
            if not text.strip():
                continue
            
            # Add formatted result
            output.append(f"# {node_id}")
            output.append(text.strip())
            output.append("")  # Empty line between results
            
        except Exception as e:
            logger.error(f"Error processing result {result}: {str(e)}")
            continue
    
    return "\n".join(output)



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
    build_parser.add_argument("--export", type=str, help="Export the built index to a compressed tar.gz file")
    build_parser.add_argument("--update", action="store_true", help="Update existing index instead of rebuilding it")
    build_parser.set_defaults(func=build_command)
    
    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve information from embeddings database")
    retrieve_parser.add_argument("embeddings", type=str, help="Path to embeddings database")
    retrieve_parser.add_argument("query", type=str, help="Search query")
    retrieve_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    retrieve_parser.add_argument("--graph", action="store_true", help="Enable graph search")
    retrieve_parser.add_argument("--min_similarity", type=float, default=0.3, help="Minimum similarity threshold for results")
    retrieve_parser.set_defaults(func=retrieve_command)
    
    # Generate command removed as it's currently unused
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.debug if hasattr(args, 'debug') else False)
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()

# Entry point functions have been moved to dedicated scripts in the bin directory

if __name__ == "__main__":
    main()
