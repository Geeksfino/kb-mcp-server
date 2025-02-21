"""Tools for the txtai MCP server."""

from .search import register_search_tools
from .text import register_text_tools

__all__ = ["register_search_tools", "register_text_tools"]
