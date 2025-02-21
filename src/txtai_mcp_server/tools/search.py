"""
Search-related tools for the txtai MCP server.
"""
import logging
import sys
from typing import Dict, List, Optional
import uuid

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Ensure logs go to stderr
)

logger = logging.getLogger(__name__)

from mcp.server.fastmcp import FastMCP, Context
from txtai.embeddings import Embeddings

from ..core.context import TxtAIContext


def register_search_tools(mcp: FastMCP) -> None:
    """Register search-related tools with the MCP server."""
    logger.debug("Starting registration of search tools...")
    
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
        if not hasattr(ctx, 'server') or not hasattr(ctx.server, 'lifespan_context'):
            raise RuntimeError("Server not properly initialized - lifespan context is missing")
            
        lifespan_context = ctx.server.lifespan_context
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
        if not hasattr(ctx, 'server') or not hasattr(ctx.server, 'lifespan_context'):
            raise RuntimeError("Server not properly initialized - lifespan context is missing")
            
        lifespan_context = ctx.server.lifespan_context
        if not lifespan_context or "txtai_context" not in lifespan_context:
            raise RuntimeError("TxtAI context not initialized")
            
        txtai_context: TxtAIContext = lifespan_context["txtai_context"]
        data = [(id or str(uuid.uuid4()), content)]
        txtai_context.embeddings.upsert(data)
        return {"message": "Content added successfully"}
    
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
        if not hasattr(ctx, 'server') or not hasattr(ctx.server, 'lifespan_context'):
            raise RuntimeError("Server not properly initialized - lifespan context is missing")
            
        lifespan_context = ctx.server.lifespan_context
        if not lifespan_context or "txtai_context" not in lifespan_context:
            raise RuntimeError("TxtAI context not initialized")
            
        txtai_context: TxtAIContext = lifespan_context["txtai_context"]
        txtai_context.embeddings.delete([id])
        return {"message": "Content deleted successfully"}
    
    logger.debug("Search tools registered successfully.")
