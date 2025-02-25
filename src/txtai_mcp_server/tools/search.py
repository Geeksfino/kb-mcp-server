"""
Search-related tools for the txtai MCP server.
"""
import logging
import sys
import traceback
import uuid
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import Field

from ..core.context import TxtAIContext
from ..core.state import get_txtai_app

logger = logging.getLogger(__name__)

def get_txtai_context(ctx: Context) -> TxtAIContext:
    """Helper to get TxtAI context with error handling."""
    request_context = ctx.request_context
    if not request_context or not hasattr(request_context, 'lifespan_context'):
        raise RuntimeError("Server not properly initialized - request context or lifespan context is missing")
    
    lifespan_context = request_context.lifespan_context
    if not isinstance(lifespan_context, TxtAIContext):
        raise RuntimeError(f"Invalid lifespan context type: {type(lifespan_context)}")
    
    return lifespan_context

def register_search_tools(mcp: FastMCP) -> None:
    """Register search-related tools with the MCP server."""
    logger.debug("Starting registration of search tools...")
    
    @mcp.tool(
        name="semantic_search",
        description="""Find documents or passages that are semantically similar to the query using AI embeddings.
        Best used for:
        - Finding relevant documents based on meaning/concepts
        - Getting background information for answering questions
        - Discovering content related to a topic
        - Searching using natural language questions
        
        Uses hybrid search to combine semantic understanding with keyword matching.
        
        Example: "What are the best practices for error handling?" will find documents about error handling patterns."""
    )
    async def semantic_search(
        ctx: Context,
        query: str,
        limit: Optional[int] = Field(5, description="Maximum number of results to return"),
        threshold: Optional[float] = Field(None, description="Minimum similarity score threshold"),
    ) -> List[Dict]:
        """Execute semantic search using txtai."""
        logger.debug(f"Semantic search request - query: {query}, limit: {limit}, threshold: {threshold}")
        try:
            app = get_txtai_app()
            results = app.search(query, limit=limit)
            return results
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}\n{traceback.format_exc()}")
            raise

    @mcp.tool(
        name="add_content",
        description="""Add new content to the search index.
        Best used for:
        - Adding new documents or text for searching
        - Storing information for later retrieval
        - Building a knowledge base
        
        Content is immediately available for search.
        
        Example: Adding documentation, articles, or knowledge base entries."""
    )
    async def add_content(
        ctx: Context,
        text: str,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ) -> Dict:
        """Add content to the search index.
        
        Args:
            text: Text content to add
            metadata: Optional metadata dict
            tags: Optional list of tags
            
        Returns:
            Dict with status and id of added content
        """
        try:
            logger.debug(f"Adding content: {text[:100]}...")
            app = get_txtai_app()
            
            # Create document with metadata
            doc_id = str(uuid.uuid4())
            document = {
                "id": doc_id,
                "text": text,
                "source": "user_added",
                "tags": tags or []
            }
            if metadata:
                document.update(metadata)
            
            # Add to index using Application
            app.add([document])
            app.index()
            
            logger.info(f"Added content with id: {doc_id}")
            return {"status": "success", "id": doc_id}
            
        except Exception as e:
            logger.error(f"Failed to add content: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to add content: {str(e)}")
            
    logger.debug("Search tools registered successfully")
