"""
txtai MCP server implementation.
"""
import sys
import os
import argparse
import asyncio
from importlib.metadata import version

try:
    __version__ = version("txtai-mcp-server")
except ImportError:
    __version__ = "0.1.0"  # fallback version

from . import server

def main():
    """Main entry point for the package."""
    parser = argparse.ArgumentParser(description='TxtAI MCP Server')
    parser.add_argument('--embeddings', type=str, help='Path to embeddings directory or archive file')
    parser.add_argument('--transport', type=str, default='sse', choices=['sse', 'stdio'], 
                        help='Transport to use for MCP server (default: sse)')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host to bind to when using SSE transport (default: localhost)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to bind to when using SSE transport (default: 8000)')
    
    args = parser.parse_args()
    
    # Set environment variable if embeddings path is provided
    if args.embeddings:
        os.environ["TXTAI_EMBEDDINGS"] = args.embeddings
    
    # Import here to avoid circular imports
    from .server import run_server
    
    # Run the server with the specified transport
    if args.transport == 'sse':
        asyncio.run(run_server(transport='sse', host=args.host, port=args.port))
    else:
        asyncio.run(run_server(transport='stdio'))

__all__ = ["__version__", "main"]
