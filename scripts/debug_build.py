#!/usr/bin/env python3
"""
Debug script for tracing the kb-build process with enhanced logging.
"""

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

# Import txtai
from txtai.app import Application
from txtai.embeddings import Embeddings

# Add parent directory to path to import kb_builder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kb_builder.cli import build_command, create_application

# Configure logging
logger = logging.getLogger("kb-debug")

def setup_logging(debug: bool = False, trace: bool = False):
    """Set up enhanced logging configuration.
    
    Args:
        debug: Whether to enable debug logging
        trace: Whether to enable trace logging (even more verbose)
    """
    if trace:
        level = logging.DEBUG
        # Enable txtai internal logging
        txtai_logger = logging.getLogger("txtai")
        txtai_logger.setLevel(logging.DEBUG)
        
        # Enable other relevant loggers
        logging.getLogger("faiss").setLevel(logging.DEBUG)
        logging.getLogger("transformers").setLevel(logging.INFO)
    else:
        level = logging.DEBUG if debug else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

def inspect_embeddings(app: Application):
    """
    Inspect the embeddings object and print detailed information.
    
    Args:
        app: txtai Application instance
    """
    logger.info("--- EMBEDDINGS INSPECTION ---")
    
    if not hasattr(app, 'embeddings') or not app.embeddings:
        logger.error("No embeddings found in application!")
        return
    
    # Basic embeddings info
    logger.info(f"Embeddings path: {app.embeddings.path if hasattr(app.embeddings, 'path') else 'Not set'}")
    logger.info(f"Embeddings dimension: {app.embeddings.dimension if hasattr(app.embeddings, 'dimension') else 'Unknown'}")
    
    # Check if index is initialized
    if hasattr(app.embeddings, 'initialized'):
        logger.info(f"Embeddings initialized: {app.embeddings.initialized}")
    else:
        logger.info("Embeddings initialized status: Unknown")
    
    # Check backend
    if hasattr(app.embeddings, 'backend'):
        logger.info(f"Backend type: {type(app.embeddings.backend).__name__}")
    else:
        logger.info("Backend: Not initialized")
    
    # Check database
    if hasattr(app.embeddings, 'database'):
        logger.info(f"Database type: {type(app.embeddings.database).__name__ if app.embeddings.database else 'None'}")
        if app.embeddings.database:
            logger.info(f"Database count: {app.embeddings.count() if hasattr(app.embeddings, 'count') else 'Unknown'}")
    else:
        logger.info("Database: Not initialized")
    
    # Check storage path
    if hasattr(app.embeddings, 'config') and 'path' in app.embeddings.config:
        storage_path = app.embeddings.config['path']
        logger.info(f"Storage path: {storage_path}")
        
        # Check if path exists
        if os.path.exists(storage_path):
            logger.info(f"Storage path exists: Yes")
            # List files in storage path
            files = list(Path(storage_path).glob("*"))
            logger.info(f"Files in storage path: {[f.name for f in files]}")
        else:
            logger.info(f"Storage path exists: No")
    else:
        logger.info("Storage path: Not configured")
    
    # Check graph
    if hasattr(app.embeddings, 'graph'):
        logger.info(f"Graph initialized: {app.embeddings.graph is not None}")
        if app.embeddings.graph:
            logger.info(f"Graph type: {type(app.embeddings.graph).__name__}")
            # Try to get graph stats
            try:
                if hasattr(app.embeddings.graph, 'graph'):
                    g = app.embeddings.graph.graph
                    logger.info(f"Graph nodes: {len(g.nodes)}")
                    logger.info(f"Graph edges: {len(g.edges)}")
            except Exception as e:
                logger.error(f"Error getting graph stats: {e}")
    else:
        logger.info("Graph: Not configured")

def inspect_config(config_path: str):
    """
    Inspect the configuration file and print detailed information.
    
    Args:
        config_path: Path to configuration file
    """
    logger.info("--- CONFIG INSPECTION ---")
    
    if not config_path or not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
    
    try:
        # Determine file type
        if config_path.endswith(('.yml', '.yaml')):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            logger.error(f"Unsupported config file format: {config_path}")
            return
        
        # Print config details
        logger.info(f"Config file: {config_path}")
        
        # Check path
        if 'path' in config:
            path = config['path']
            logger.info(f"Index path: {path}")
            # Convert relative path to absolute
            if not os.path.isabs(path):
                abs_path = os.path.abspath(os.path.join(os.path.dirname(config_path), path))
                logger.info(f"Absolute index path: {abs_path}")
                # Check if path exists
                if os.path.exists(abs_path):
                    logger.info(f"Path exists: Yes")
                else:
                    logger.info(f"Path exists: No (will be created)")
            else:
                # Check if path exists
                if os.path.exists(path):
                    logger.info(f"Path exists: Yes")
                else:
                    logger.info(f"Path exists: No (will be created)")
        else:
            logger.info("Index path: Not specified")
        
        # Check embeddings config
        if 'embeddings' in config:
            emb_config = config['embeddings']
            logger.info(f"Embeddings model: {emb_config.get('path', 'Not specified')}")
            logger.info(f"Content storage: {emb_config.get('content', False)}")
            logger.info(f"Writable: {emb_config.get('writable', False)}")
            
            # Check backend
            if 'backend' in emb_config:
                logger.info(f"Backend: {emb_config['backend']}")
            else:
                logger.info("Backend: Not specified (will use default)")
            
            # Check graph config
            if 'graph' in emb_config:
                graph_config = emb_config['graph']
                logger.info(f"Graph backend: {graph_config.get('backend', 'Not specified')}")
                logger.info(f"Graph limit: {graph_config.get('limit', 'Not specified')}")
                logger.info(f"Graph min score: {graph_config.get('minscore', 'Not specified')}")
        else:
            logger.info("Embeddings config: Not specified")
        
    except Exception as e:
        logger.error(f"Error inspecting config: {e}")
        logger.error(traceback.format_exc())

def debug_export(app: Application, export_path: str):
    """
    Debug the export process and print detailed information.
    
    Args:
        app: txtai Application instance
        export_path: Path to export file
    """
    logger.info("--- EXPORT DEBUG ---")
    
    if not export_path:
        logger.error("No export path specified")
        return
    
    # Add .tar.gz extension if not present
    if not export_path.endswith('.tar.gz'):
        export_path = f"{export_path}.tar.gz"
    
    logger.info(f"Export path: {export_path}")
    
    # Check if embeddings is initialized
    if not hasattr(app, 'embeddings') or not app.embeddings:
        logger.error("No embeddings found in application!")
        return
    
    # Check if embeddings has data
    count = app.embeddings.count() if hasattr(app.embeddings, 'count') else 0
    logger.info(f"Embeddings count: {count}")
    
    # Check if save method exists
    if not hasattr(app.embeddings, 'save'):
        logger.error("Embeddings object does not have a save method!")
        return
    
    # Try to save
    try:
        logger.info(f"Calling app.embeddings.save({export_path})")
        
        # Check storage path before saving
        if hasattr(app.embeddings, 'config') and 'path' in app.embeddings.config:
            storage_path = app.embeddings.config['path']
            logger.info(f"Storage path before save: {storage_path}")
            
            # Check if path exists
            if os.path.exists(storage_path):
                logger.info(f"Storage path exists before save: Yes")
                # List files in storage path
                files = list(Path(storage_path).glob("*"))
                logger.info(f"Files in storage path before save: {[f.name for f in files]}")
            else:
                logger.info(f"Storage path exists before save: No")
        
        # Save embeddings
        app.embeddings.save(export_path)
        
        logger.info(f"Save completed successfully")
        
        # Check if export file exists
        if os.path.exists(export_path):
            logger.info(f"Export file exists: Yes (size: {os.path.getsize(export_path)} bytes)")
            
            # Try to examine tar.gz content
            try:
                import tarfile
                with tarfile.open(export_path, 'r:gz') as tar:
                    members = tar.getmembers()
                    logger.info(f"Tar.gz contains {len(members)} files:")
                    for member in members:
                        logger.info(f"  - {member.name} ({member.size} bytes)")
            except Exception as e:
                logger.error(f"Error examining tar.gz content: {e}")
        else:
            logger.error(f"Export file does not exist after save!")
    except Exception as e:
        logger.error(f"Error during save: {e}")
        logger.error(traceback.format_exc())

def debug_build_command(args):
    """
    Enhanced debug version of the build command.
    
    Args:
        args: Command-line arguments
    """
    logger.info("=== STARTING DEBUG BUILD COMMAND ===")
    
    # Inspect config
    if hasattr(args, 'config') and args.config:
        inspect_config(args.config)
    
    # Create application
    logger.info("Creating application...")
    app = create_application(args.config if hasattr(args, 'config') else None)
    
    # Inspect embeddings before processing
    logger.info("Inspecting embeddings before processing...")
    inspect_embeddings(app)
    
    # Process documents (simplified version)
    logger.info("Processing documents...")
    try:
        # Call the original build command to process documents
        build_command(args)
        
        # Re-create application to ensure it's up-to-date
        logger.info("Re-creating application after build...")
        app = create_application(args.config if hasattr(args, 'config') else None)
        
        # Inspect embeddings after processing
        logger.info("Inspecting embeddings after processing...")
        inspect_embeddings(app)
        
        # Debug export if requested
        if hasattr(args, 'export') and args.export:
            debug_export(app, args.export)
            
    except Exception as e:
        logger.error(f"Error in debug build command: {e}")
        logger.error(traceback.format_exc())
    
    logger.info("=== DEBUG BUILD COMMAND COMPLETED ===")

def main():
    """
    Main entry point for the debug build script.
    """
    # Create parser
    parser = argparse.ArgumentParser(prog='debug-kb-build', description="Debug build embeddings database with enhanced logging")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--trace", action="store_true", help="Enable trace logging (even more verbose)")
    parser.add_argument("--input", type=str, nargs="+", help="Path to input files or directories")
    parser.add_argument("--extensions", type=str, help="Comma-separated list of file extensions to include")
    parser.add_argument("--json_input", type=str, help="Path to JSON file containing a list of documents")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--export", type=str, help="Export the built index to a compressed tar.gz file")
    parser.add_argument("--update", action="store_true", help="Update existing index instead of rebuilding it")
    
    args = parser.parse_args()
    
    # Set up enhanced logging
    setup_logging(args.debug, args.trace)
    
    # Run debug build command
    debug_build_command(args)

if __name__ == "__main__":
    main()
