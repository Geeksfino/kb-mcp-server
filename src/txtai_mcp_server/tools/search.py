"""
Search-related tools for the txtai MCP server.
"""
import logging
import sys
from typing import Dict, List, Optional
import uuid
import traceback

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
        try:
            logger.debug("Executing semantic_search tool...")
            request_context = ctx.request_context
            if not request_context or not hasattr(request_context, 'lifespan_context'):
                raise RuntimeError("Server not properly initialized - request context or lifespan context is missing")
            
            lifespan_context = request_context.lifespan_context
            if not lifespan_context or "txtai_context" not in lifespan_context:
                raise RuntimeError("TxtAI context not initialized")
            
            txtai_context: TxtAIContext = lifespan_context["txtai_context"]
            logger.debug(f"Searching with query: {query}, limit: {limit}")
            
            # Use SQL-like search with content fields
            sql = f"select text, context, answer, score from txtai where similar('{query}') limit {limit}"
            all_results = txtai_context.embeddings.search(sql)
            
            logger.info(f"Search results ({len(all_results)} total):")
            for r in all_results[:10]:  # Show top 10 for debugging
                logger.info(f"  Score: {r['score']:.3f}")
                logger.info(f"  Question: {r['text']}")
                if r.get('context'):  # Only show context if it exists
                    logger.info(f"  Context: {r['context'][:200]}...")
                if r.get('answer'):
                    logger.info(f"  Answer: {r['answer']}\n")
            
            # Process results into standard format
            processed_results = [
                {
                    "id": f"result_{i}",
                    "score": float(r["score"]),
                    "content": r.get("context", r["text"]),  # Use context if available, otherwise question
                    "question": r["text"],
                    "answer": r.get("answer", "")  # Handle missing answer field
                }
                for i, r in enumerate(all_results)
            ]
            
            logger.debug(f"Processed {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in semantic_search: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    @mcp.tool()
    async def add_content(
        ctx: Context,
        content: str,
        id: Optional[str] = None
    ) -> None:
        """
        Add content to the search index.
        
        Args:
            content: Content to add
            id: Optional identifier for the content
        """
        try:
            logger.debug("Executing add_content tool...")
            request_context = ctx.request_context
            if not request_context or not hasattr(request_context, 'lifespan_context'):
                raise RuntimeError("Server not properly initialized - request context or lifespan context is missing")
            
            lifespan_context = request_context.lifespan_context
            if not lifespan_context or "txtai_context" not in lifespan_context:
                raise RuntimeError("TxtAI context not initialized")
            
            txtai_context: TxtAIContext = lifespan_context["txtai_context"]
            content_id = id or str(uuid.uuid4())
            logger.debug(f"Adding content with id: {content_id}")
            
            # Create metadata for the content
            metadata = {
                "text": content,  # Store original text
                "source": "user_added"  # Mark as user-added content
            }
            
            # Create a list of (id, metadata, tags) tuples for indexing
            data = [(content_id, metadata, None)]
            
            # Get existing index size for debugging
            existing_count = len(txtai_context.embeddings.search("select id from txtai limit 1"))
            logger.debug(f"Current index size before adding: {existing_count}")
            
            # Append to existing index
            txtai_context.embeddings.upsert(data)
            
            # Verify addition
            new_count = len(txtai_context.embeddings.search("select id from txtai limit 1"))
            logger.debug(f"New index size after adding: {new_count}")
            
            logger.debug("Content added successfully")
            return {"message": "Content added successfully", "id": content_id}
        except Exception as e:
            logger.error(f"Error in add_content: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    @mcp.tool()
    async def delete_content(
        ctx: Context,
        id: str
    ) -> None:
        """
        Delete content from the search index.
        
        Args:
            id: Identifier of the content to delete
        """
        try:
            logger.debug("Executing delete_content tool...")
            request_context = ctx.request_context
            if not request_context or not hasattr(request_context, 'lifespan_context'):
                raise RuntimeError("Server not properly initialized - request context or lifespan context is missing")
            
            lifespan_context = request_context.lifespan_context
            if not lifespan_context or "txtai_context" not in lifespan_context:
                raise RuntimeError("TxtAI context not initialized")
            
            txtai_context: TxtAIContext = lifespan_context["txtai_context"]
            logger.debug(f"Deleting content with id: {id}")
            txtai_context.embeddings.delete([id])
            logger.debug("Content deleted successfully")
            return {"message": "Content deleted successfully"}
        except Exception as e:
            logger.error(f"Error in delete_content: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    logger.debug("Search tools registered successfully.")
