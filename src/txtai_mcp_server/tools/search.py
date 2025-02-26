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
        
        Example: "What are the best practices for error handling?" will find documents about error handling patterns."""
    )
    async def semantic_search(
        ctx: Context,
        query: str,
        limit: Optional[int] = Field(5, description="Maximum number of results to return"),
    ) -> str:
        """Execute semantic search using txtai."""
        logger.info(f"Semantic search request - query: {query}, limit: {limit}")
        try:
            app = get_txtai_app()
            # Debug embeddings state
            logger.info(f"Embeddings config: {app.config.get('embeddings')}")
            
            # Get search results
            results = app.search(query, limit=limit)
            logger.info(f"Search results (raw): {results}")
            
            # If no results, return empty list
            if not results:
                logger.info("No search results found")
                return "[]"
            
            # Format results to match the expected format
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
        name="search",
        description="Search for documents in the index."
    )
    async def search(
        ctx: Context,
        query: str,
        limit: int = 3
    ) -> Dict:
        """Search for documents in the index.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
        """
        try:
            logger.info(f"Searching for: {query} (limit: {limit})")
            app = get_txtai_app()
            
            # For direct ID lookups, handle specially
            if "where id = " in query:
                try:
                    import re
                    match = re.search(r"where id = '([^']*)'", query)
                    if match:
                        doc_id = match.group(1)
                        # Try cache first
                        from txtai_mcp_server.core.state import get_document_from_cache
                        text = get_document_from_cache(doc_id)
                        if text:
                            return [{"id": doc_id, "text": text, "score": 1.0}]
                        
                        # If not in cache, try SQL with proper escaping
                        safe_id = escape_sql_string(doc_id)
                        # Use a more specific SQL query that ensures we get the exact ID match
                        sql_query = f"select id, text from txtai where id = '{safe_id}' and text is not null"
                        results = app.search(sql_query)
                        if results and len(results) > 0:
                            result = results[0]
                            if isinstance(result, dict) and result.get("id") == doc_id:
                                # Add to cache for next time
                                add_document_to_cache(doc_id, result.get("text"))
                                return [result]
                        
                        # If SQL fails, try backend directly
                        if hasattr(app.embeddings, "backend"):
                            try:
                                doc = app.embeddings.backend.get(doc_id)
                                if doc:
                                    return [{"id": doc_id, "text": doc, "score": 1.0}]
                            except Exception as e:
                                logger.warning(f"Backend lookup failed: {e}")
                except Exception as e:
                    logger.warning(f"ID lookup failed: {e}")
                    # Don't return empty list yet, try semantic search as fallback
            
            # For wildcard searches, try multiple approaches
            if query == "*":
                results = set()
                
                # Try backend IDs first
                if hasattr(app.embeddings, "backend"):
                    try:
                        backend_ids = app.embeddings.backend.ids()
                        logger.info(f"Found {len(backend_ids)} IDs in backend")
                        for doc_id in backend_ids:
                            try:
                                # Try to get document from cache first
                                text = get_document_from_cache(doc_id)
                                if text:
                                    results.add((doc_id, text))
                                    continue
                                
                                # If not in cache, try backend
                                doc = app.embeddings.backend.get(doc_id)
                                if doc:
                                    results.add((doc_id, doc))
                            except Exception as e:
                                logger.warning(f"Failed to get document {doc_id}: {e}")
                    except Exception as e:
                        logger.warning(f"Backend ID lookup failed: {e}")
                
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
            
            # For normal searches, use the embeddings search
            results = app.search(query, limit)
            return results
            
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to search: {str(e)}")

    @mcp.tool(
        name="add_content",
        description="[DEPRECATED] Add content to the search index. Use add_documents instead."
    )
    async def add_content(
        ctx: Context,
        text: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """[DEPRECATED] Add content to the search index.
        
        This tool is deprecated. Please use add_documents instead.
        
        Args:
            text: The text content to add
            metadata: Optional metadata to associate with the content
        """
        try:
            logger.warning("add_content is deprecated. Please use add_documents instead.")
            
            # Generate a document ID if not provided in metadata
            doc_id = metadata.get("id") if metadata and "id" in metadata else str(uuid.uuid4())
            
            # Create a single document and use add_documents
            document = {
                "id": doc_id,
                "text": text
            }
            
            if metadata:
                document["metadata"] = metadata
            
            # Call add_documents with a single document
            result = await add_documents(ctx, [document])
            
            # Return a compatible result
            return {
                "status": "success",
                "id": doc_id,
                "message": f"Added content with ID: {doc_id}"
            }
            
        except Exception as e:
            logger.error(f"Failed to add content: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to add content: {str(e)}")

    @mcp.tool(
        name="add_documents",
        description="Add documents to the search index."
    )
    async def add_documents(
        ctx: Context,
        documents: List[Dict[str, str]]
    ) -> Dict:
        """Add documents to the search index.
        
        Args:
            documents: List of documents to add, each with 'id' and 'text' fields
        """
        try:
            logger.info(f"Adding {len(documents)} documents to index")
            app = get_txtai_app()
            
            # Add documents to txtai
            formatted_docs = []
            for doc in documents:
                doc_id = doc.get("id")
                text = doc.get("text")
                if doc_id and text:
                    # Format as dictionary for txtai Application
                    formatted_docs.append({"id": doc_id, "text": text})
                    # Add to cache
                    from txtai_mcp_server.core.state import add_document_to_cache
                    add_document_to_cache(doc_id, text)
            
            # Add documents and build the index
            app.add(formatted_docs)
            app.index()
            
            return {
                "status": "success",
                "count": len(documents),
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
