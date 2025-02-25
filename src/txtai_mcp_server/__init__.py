"""
txtai MCP server implementation.
"""
from importlib.metadata import version

try:
    __version__ = version("txtai-mcp-server")
except ImportError:
    __version__ = "0.1.0"  # fallback version

__all__ = ["__version__"]
