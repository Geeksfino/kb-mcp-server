"""
Search-related tools for the txtai MCP server.
"""
import logging
import sys
import traceback
import uuid
import json
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import Field

from ..core.context import TxtAIContext
from ..core.state import get_txtai_app, get_document_cache, add_to_document_cache, get_document_from_cache

logger = logging.getLogger(__name__)

def get_txtai_context(ctx: Context) -> TxtAIContext:
    """Helper to get TxtAI context with error handling."""
    request_context = ctx.request_context
    if not request_context or not request_context.lifespan_context:
        raise RuntimeError("Server not properly initialized - request context or lifespan context is missing")
    
    lifespan_context = request_context.lifespan_context
    if not isinstance(lifespan_context, TxtAIContext):
        raise RuntimeError(f"Invalid lifespan context type: {type(lifespan_context)}")
    
    return lifespan_context

def escape_sql_string(text: str) -> str:
    """Escape a string for use in SQL queries."""
    if text is None:
        return text
    return text.replace("'", "''")

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
        
        When graph=True, performs graph-based search to find relationships between documents.
        
        Example: "What are the best practices for error handling?" will find documents about error handling patterns."""
    )
    async def semantic_search(
        ctx: Context,
        query: str,
        limit: Optional[int] = Field(5, description="Maximum number of results to return"),
        graph: Optional[bool] = Field(True, description="Enable graph-based search to find relationships between documents"),
    ) -> str:
        """Execute semantic search using txtai."""
        logger.info(f"Semantic search request - query: {query}, limit: {limit}, graph: {graph}")
        try:
            app = get_txtai_app()
            # Debug embeddings state
            logger.info(f"Embeddings config: {app.config.get('embeddings')}")
            
            # Get search results with graph parameter
            results = app.search(query, limit=limit, graph=graph)
            logger.info(f"Search results (raw): {results}")
            
            # If no results, return empty list
            if not results:
                logger.info("No search results found")
                return "[]"
            
            # Handle different result types based on graph parameter
            if graph:
                # For graph search, results have a different structure
                formatted_results = []
                
                # Process centrality nodes from graph search
                for node_id in list(results.centrality().keys())[:limit]:
                    node_data = results.node(node_id)
                    
                    # Format node data for API response
                    formatted_node = {
                        "id": node_id,
                        "text": node_data.get("text", "No text available"),
                        "score": node_data.get("score", 0.0),
                        "centrality": results.centrality()[node_id],
                        "connections": len(results.neighbors(node_id))
                    }
                    
                    formatted_results.append(formatted_node)
                
                # Return formatted graph results as JSON
                return json.dumps(formatted_results)
            else:
                # Format results for regular search
                formatted_results = []
                
                # Get the global document cache
                document_cache = get_document_cache()
                logger.info(f"Document cache size: {len(document_cache)}, keys: {list(document_cache.keys())}")
                
                for result in results:
                    # Application.search() returns [{"id": id, "score": score}]
                    if isinstance(result, dict) and "id" in result and "score" in result:
                        doc_id = result["id"]
                        score = result["score"]
                        
                        # Try to get the document text from the cache
                        text = get_document_from_cache(doc_id) or "No text available"
                        logger.info(f"Retrieved document {doc_id}: text available: {text != 'No text available'}")
                        
                        # Add formatted result
                        formatted_results.append({
                            "id": doc_id,
                            "score": score,
                            "text": text
                        })
                
                # Return formatted results as JSON
                return json.dumps(formatted_results)
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}\n{traceback.format_exc()}")
            raise

    @mcp.tool(
        name="add_documents",
        description="Add multiple documents to the search index."
    )
    async def add_documents(
        ctx: Context,
        documents: List[Dict]
    ) -> Dict:
        """Add multiple documents to the search index.
        
        Args:
            documents: List of document objects with at least "id" and "text" fields
        """
        try:
            # Get txtai app
            app = get_txtai_app()
            
            # Validate documents
            valid_documents = []
            for doc in documents:
                if not isinstance(doc, dict):
                    logger.warning(f"Skipping invalid document (not a dict): {doc}")
                    continue
                    
                if "id" not in doc or "text" not in doc:
                    logger.warning(f"Skipping document missing required fields: {doc}")
                    continue
                
                valid_documents.append(doc)
                
                # Add to document cache
                add_to_document_cache(doc["id"], doc["text"])
            
            if not valid_documents:
                return {
                    "status": "error",
                    "message": "No valid documents provided"
                }
            
            # Add documents to index
            logger.info(f"Adding {len(valid_documents)} documents to index")
            app.add(valid_documents)
            
            # Save index
            if app.config.get("path"):
                logger.info(f"Saving index to: {app.config.get('path')}")
                app.index()
            
            return {
                "status": "success",
                "count": len(valid_documents),
                "ids": [doc["id"] for doc in valid_documents]
            }
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to add documents: {str(e)}")

    @mcp.tool(
        name="list_documents",
        description="List all documents in the search index."
    )
    async def list_documents(
        ctx: Context,
        limit: int = 100
    ) -> Dict:
        """List all documents in the search index.
        
        Args:
            limit: Maximum number of documents to return
        """
        try:
            logger.info(f"Listing documents (limit: {limit})")
            app = get_txtai_app()
            
            # Get documents from both the embeddings backend and cache
            documents = []
            doc_ids = set()  # Track unique document IDs
            
            # Try to get documents from embeddings backend
            if hasattr(app.embeddings, "backend"):
                try:
                    # Get document IDs from backend
                    backend_ids = app.embeddings.backend.ids()
                    logger.info(f"Found {len(backend_ids)} document IDs in backend")
                    doc_ids.update(backend_ids)
                except Exception as e:
                    logger.warning(f"Error getting IDs from backend: {e}")
            
            # Try to get documents using SQL
            try:
                sql_results = app.search("select id, text from txtai")
                logger.info(f"SQL query found {len(sql_results)} documents")
                for result in sql_results:
                    if isinstance(result, dict):
                        doc_id = result.get("id")
                        if doc_id:
                            doc_ids.add(doc_id)
            except Exception as e:
                logger.warning(f"SQL query failed: {e}")
            
            # Try wildcard search as well
            try:
                search_results = app.search("*", limit)
                logger.info(f"Wildcard search found {len(search_results)} documents")
                for result in search_results:
                    if isinstance(result, dict):
                        doc_id = result.get("id")
                        if doc_id:
                            doc_ids.add(doc_id)
                    elif isinstance(result, (list, tuple)) and len(result) > 0:
                        doc_id = result[0]
                        if doc_id:
                            doc_ids.add(doc_id)
            except Exception as e:
                logger.warning(f"Wildcard search failed: {e}")
            
            # Now get the text for each unique document ID
            logger.info(f"Found {len(doc_ids)} unique document IDs")
            for doc_id in list(doc_ids)[:limit]:
                text = get_document_from_cache(doc_id)
                if text:
                    documents.append({
                        "id": doc_id,
                        "score": 1.0,  # Default score since we're not doing similarity search
                        "text": text[:100] + "..." if len(text) > 100 else text
                    })
            
            return {
                "status": "success",
                "count": len(documents),
                "documents": documents
            }
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to list documents: {str(e)}")
            
    logger.debug("Search tools registered successfully")
