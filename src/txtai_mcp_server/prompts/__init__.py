"""Prompts for the txtai MCP server."""

from .search import register_search_prompts
from .analysis import register_analysis_prompts

__all__ = ["register_search_prompts", "register_analysis_prompts"]
