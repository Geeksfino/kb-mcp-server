#!/usr/bin/env python3
"""
Entry point script for kb-build command.
"""

import argparse
import logging
import sys

from kb_builder.cli import build_command

def main():
    """
    Main entry point for kb-build command.
    """
    # Set up logging
    def setup_logging(debug=False):
        level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )

    # Create a parser specifically for the build command
    parser = argparse.ArgumentParser(prog='kb-build', description="Build embeddings database")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--input", type=str, nargs="+", help="Path to input files or directories")
    parser.add_argument("--extensions", type=str, help="Comma-separated list of file extensions to include")
    parser.add_argument("--json_input", type=str, help="Path to JSON file containing a list of documents")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--export", type=str, help="Export the built index to a compressed tar.gz file")
    parser.add_argument("--update", action="store_true", help="Update existing index instead of rebuilding it")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.debug)
    
    # Call the build command directly
    build_command(args)

if __name__ == "__main__":
    main()
