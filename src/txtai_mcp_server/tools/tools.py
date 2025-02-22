"""
Search-related tools for the txtai MCP server.
"""
import logging
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP, Context

logger = logging.getLogger(__name__)

def register_search_tools(mcp: FastMCP) -> None:
    """Register search-related tools."""
    logger.debug("Registering search tools...")
    
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
            limit: Maximum number of results
        """
        logger.debug(f"Executing search: query='{query}', limit={limit}")
        
        if not ctx.lifespan_context or "txtai_context" not in ctx.lifespan_context:
            raise RuntimeError("TxtAI context not initialized")
            
        txtai_context = ctx.lifespan_context["txtai_context"]
        try:
            results = txtai_context.embeddings.search(query, limit=limit)
            return [{"score": float(score), "content": content} 
                   for score, content in results]
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    @mcp.tool()
    async def add_content(
        ctx: Context,
        content: str,
        id: Optional[str] = None
    ) -> Dict:
        """Add content to search index."""
        logger.debug(f"Adding content: id='{id}'")
        
        if not ctx.lifespan_context or "txtai_context" not in ctx.lifespan_context:
            raise RuntimeError("TxtAI context not initialized")
            
        txtai_context = ctx.lifespan_context["txtai_context"]
        try:
            if id:
                txtai_context.embeddings.add([(id, content)])
            else:
                txtai_context.embeddings.add(content)
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Add content error: {e}")
            raise

    logger.info("Search tools registered successfully")