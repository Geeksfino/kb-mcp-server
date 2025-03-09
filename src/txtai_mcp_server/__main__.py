"""
Command-line entry point for the txtai MCP server with causal boost support.

This module enables running the server directly with all causal boost features:

1. Language Support:
   - Automatic detection of English, Spanish, and French queries
   - Language-specific causal patterns and keywords
   - Configurable fallback behavior

2. Causal Boost Features:
   - Dynamic relevance scoring:
     * 1.3x boost for explicit causal queries
     * 1.2x boost for multiple causal terms
     * 1.1x boost for general queries
     * 0.7x penalty for negated causality
   - Domain-specific terminology support
   - Negation pattern detection

3. Configuration:
   - YAML configuration files for patterns
   - Environment variables (TXTAI_CAUSAL_CONFIG)
   - Command-line arguments for quick setup

Usage:
    python -m txtai_mcp_server \
        --enable-causal-boost \
        --causal-config /path/to/config.yaml

Note:
    This module uses the server.run() function directly to avoid
    redundant imports through __init__.py, while maintaining all
    causal boost functionality.
"""
from .server import run

if __name__ == "__main__":
    run()
