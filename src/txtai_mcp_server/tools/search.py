"""
Search-related tools for the txtai MCP server.
"""
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP, Context
from txtai.embeddings import Embeddings

from ..core.context import TxtAIContext


def register_search_tools(mcp: FastMCP) -> None:
    """Register search-related tools with the MCP server."""
    
    @mcp.tool()
    async def semantic_search(
        ctx: Context,
        query: str,
        limit: Optional[int] = 10
    ) -> List[Dict]:
        """
        Search for semantically similar content.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
        """
        lifespan_context = ctx.lifespan_context
        if not lifespan_context or "txtai_context" not in lifespan_context:
            raise RuntimeError("TxtAI context not initialized")
            
        txtai_context: TxtAIContext = lifespan_context["txtai_context"]
        results = txtai_context.embeddings.search(query, limit=limit)
        return [
            {
                "score": float(score),
                "content": content
            }
            for score, content in results
        ]
    
    @mcp.tool()
    async def add_content(
        ctx: Context,
        content: str,
        id: Optional[str] = None
    ) -> Dict:
        """
        Add content to the search index.
        
        Args:
            content: Content to add
            id: Optional identifier for the content
        """
        lifespan_context = ctx.lifespan_context
        if not lifespan_context or "txtai_context" not in lifespan_context:
            raise RuntimeError("TxtAI context not initialized")
            
        txtai_context: TxtAIContext = lifespan_context["txtai_context"]
        txtai_context.embeddings.add(content)
        return {"status": "success", "message": "Content added to index"}
    
    @mcp.tool()
    async def delete_content(
        ctx: Context,
        id: str
    ) -> Dict:
        """
        Delete content from the search index.
        
        Args:
            id: Identifier of the content to delete
        """
        lifespan_context = ctx.lifespan_context
        if not lifespan_context or "txtai_context" not in lifespan_context:
            raise RuntimeError("TxtAI context not initialized")
            
        txtai_context: TxtAIContext = lifespan_context["txtai_context"]
        txtai_context.embeddings.delete([id])
        return {"status": "success"}
