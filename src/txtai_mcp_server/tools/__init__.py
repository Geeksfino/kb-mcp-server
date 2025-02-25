"""Tools for the txtai MCP server."""

from .search import register_search_tools
from .qa import register_qa_tools

__all__ = [
    "register_search_tools",
    "register_qa_tools"
]
