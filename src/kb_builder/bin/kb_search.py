#!/usr/bin/env python3
"""
Entry point script for kb-search command.
"""

import argparse
import logging
import sys

from kb_builder.cli import retrieve_command

def main():
    """
    Main entry point for kb-search command.
    """
    # Set up logging
    def setup_logging(debug=False):
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )

    # Create a parser specifically for the retrieve command
    parser = argparse.ArgumentParser(prog='kb-search', description="Search embeddings database")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("embeddings", type=str, help="Path to embeddings database")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    parser.add_argument("--graph", action="store_true", help="Enable graph search")
    parser.add_argument("--min_similarity", type=float, default=0.3, help="Minimum similarity threshold for results")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.debug)
    
    # Call the retrieve command directly
    retrieve_command(args)

if __name__ == "__main__":
    main()
