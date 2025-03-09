"""
txtai MCP server implementation with causal boost and multilingual support.

This package provides a FastMCP server with advanced causal relevance features:

1. Multilingual Support:
   - Automatic language detection for English, Spanish, and French
   - Language-specific causal patterns and keywords
   - Graceful fallback to English when needed
   - Configurable via YAML or environment variables

2. Dynamic Boost Weighting:
   - Strong boost (1.3x) for explicit causal queries
   - Medium boost (1.2x) for multiple causal terms
   - Mild boost (1.1x) for general queries
   - Penalty (0.7x) for negated causality

3. Configuration Options:
   - YAML-based pattern configuration
   - Environment variable support (TXTAI_CAUSAL_CONFIG)
   - Domain-specific terminology customization
   - Runtime parameter overrides

Usage:
    # As a module (recommended):
    python -m txtai_mcp_server --enable-causal-boost --causal-config config.yaml

    # Via MCP:
    TXTAI_CAUSAL_CONFIG=/path/to/config.yaml \
    mcp run --transport sse server.py:create_server

    # Programmatically:
    from txtai_mcp_server import create_server
    server = create_server(enable_causal_boost=True, causal_config_path="config.yaml")
"""
import sys
import os
from importlib.metadata import version

try:
    __version__ = version("txtai-mcp-server")
except ImportError:
    __version__ = "0.1.0"  # fallback version

def main():
    """Main entry point for the package.
    
    Provides command-line access to all causal boost features:
    1. Language Detection:
       - Automatic detection of English, Spanish, French
       - Smart fallback to default language
       - Language-specific pattern matching
    
    2. Causal Relevance:
       - Dynamic boost weights (1.1x-1.3x)
       - Negation detection ("not cause", "no evidence")
       - Multiple causal term detection
    
    3. Configuration:
       - YAML configuration files
       - Environment variables
       - Command-line arguments
       - Domain-specific customization
    
    Example:
        python -m txtai_mcp_server \
            --enable-causal-boost \
            --causal-config /path/to/config.yaml
    """
    from .server import run
    run()

# Export the factory function and run function for MCP compatibility
from .server import create_server, run

__all__ = ["__version__", "main", "create_server", "run"]
