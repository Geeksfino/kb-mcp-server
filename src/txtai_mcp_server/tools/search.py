"""
Search-related tools for the txtai MCP server.
"""
import logging
import sys
from typing import Dict, List, Optional
import uuid
import traceback
import re

from mcp.server.fastmcp import FastMCP, Context
from pydantic import Field

from ..core.context import TxtAIContext

logger = logging.getLogger(__name__)


def get_txtai_context(ctx: Context) -> TxtAIContext:
    """Helper to get TxtAI context with error handling."""
    request_context = ctx.request_context
    if not request_context or not hasattr(request_context, 'lifespan_context'):
        raise RuntimeError("Server not properly initialized - request context or lifespan context is missing")
    
    lifespan_context = request_context.lifespan_context
    if not lifespan_context or "txtai_context" not in lifespan_context:
        raise RuntimeError("TxtAI context not initialized")
    
    return lifespan_context["txtai_context"]


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
        limit: Optional[int] = 5,
        min_score: float = 0.3
    ) -> List[Dict]:
        """Implementation of semantic search using txtai application.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            min_score: Minimum score threshold (0-1) for results
            
        Returns:
            List of dicts containing id, score, text and metadata
        """
        try:
            logger.debug(f"Searching with query: {query}, limit: {limit}")
            txtai_context = get_txtai_context(ctx)
            
            # Use Application's search with SQL query
            sql = """
                SELECT text, answer, score, source, tags 
                FROM txtai 
                WHERE similar(?) AND score >= ? 
                ORDER BY score DESC 
                LIMIT ?
            """
            results = txtai_context.app.search(sql, (query, min_score, limit))
            
            logger.info(f"Search results ({len(results)} total):")
            for r in results[:min(10, len(results))]:
                logger.info(f"  Score: {r['score']:.3f}")
                logger.info(f"  Text: {r['text']}")
                if r.get('answer'):
                    logger.info(f"  Answer: {r['answer']}")
                if r.get('tags'):
                    logger.info(f"  Tags: {r['tags']}")
                logger.info("")
            
            processed_results = [
                {
                    "id": f"result_{i}",
                    "score": float(r["score"]),
                    "text": r["text"],
                    "answer": r.get("answer"),
                    "source": r.get("source", "unknown"),
                    "tags": r.get("tags", [])
                }
                for i, r in enumerate(results)
            ]
            
            logger.debug(f"Processed {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Search failed: {str(e)}")
    
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
            txtai_context = get_txtai_context(ctx)
            
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
            txtai_context.app.add([document])
            txtai_context.app.index()
            
            logger.info(f"Added content with id: {doc_id}")
            return {"status": "success", "id": doc_id}
            
        except Exception as e:
            logger.error(f"Failed to add content: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to add content: {str(e)}")
            
    logger.debug("Search tools registered successfully")
