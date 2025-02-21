"""
Search-related prompts for the txtai MCP server.
"""
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import PromptMessage, TextContent


def register_search_prompts(mcp: FastMCP) -> None:
    """Register search-related prompts with the MCP server."""
    
    @mcp.prompt()
    def semantic_search_prompt(query: str, context: Optional[str] = None) -> List[PromptMessage]:
        """
        Create a prompt for semantic search with optional context.
        
        Args:
            query: Search query
            context: Optional context to guide the search
        """
        messages = []
        
        # Add system message
        messages.append(
            PromptMessage(
                role="system",
                content=TextContent(
                    type="text",
                    text="You are a search assistant helping to find relevant information."
                )
            )
        )
        
        # Add context if provided
        if context:
            messages.append(
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"Consider this context while searching: {context}"
                    )
                )
            )
        
        # Add search query
        messages.append(
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"Please help me find information about: {query}"
                )
            )
        )
        
        return messages
    
    @mcp.prompt()
    def search_results_analysis(results: List[Dict], query: str) -> List[PromptMessage]:
        """
        Create a prompt to analyze search results.
        
        Args:
            results: List of search results with scores and content
            query: Original search query
        """
        # Format results for display
        formatted_results = "\n".join(
            f"Score: {r['score']:.2f}\nContent: {r['content']}\n"
            for r in results
        )
        
        messages = [
            PromptMessage(
                role="system",
                content=TextContent(
                    type="text",
                    text="You are an analyst helping to interpret search results."
                )
            ),
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"Original query: {query}\n\nSearch results:\n{formatted_results}\n\nPlease analyze these results and explain their relevance to my query."
                )
            )
        ]
        
        return messages
