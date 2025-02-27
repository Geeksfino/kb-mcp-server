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
                        
                        # Try backend directly first since it's most reliable
                        if hasattr(app.embeddings, "backend"):
                            try:
                                doc = app.embeddings.backend.get(doc_id)
                                if doc:
                                    logger.info(f"Found document {doc_id} in backend")
                                    # Handle case where doc might be a dict or other format
                                    doc_text = doc.get('text') if isinstance(doc, dict) else str(doc)
                                    # Add to cache for next time
                                    add_to_document_cache(doc_id, doc_text)
                                    return [{"id": doc_id, "text": doc_text, "score": 1.0}]
                                else:
                                    logger.warning(f"Document {doc_id} not found in backend")
                            except Exception as e:
                                logger.warning(f"Backend lookup failed for {doc_id}: {str(e)}")
                                logger.debug(f"Backend lookup traceback: {traceback.format_exc()}")
                        
                        # If backend fails, try SQL with proper escaping
                        safe_id = escape_sql_string(doc_id)
                        # Use select * to get all fields and add debug logging
                        sql_query = f"select * from txtai where id = '{safe_id}'"
                        logger.info(f"Executing SQL query: {sql_query}")
                        results = app.search(sql_query)
                        logger.info(f"SQL query results: {results}")
                        
                        if results and len(results) > 0:
                            result = results[0]
                            logger.info(f"First result: {result}")
                            if isinstance(result, dict):
                                found_id = result.get("id")
                                logger.info(f"Found ID: {found_id}, Expected ID: {doc_id}")
                                if found_id == doc_id:
                                    # Add to cache for next time
                                    add_to_document_cache(doc_id, result.get("text"))
                                    return [result]
                            else:
                                logger.warning(f"Result not in dict format: {result}")
                        
                        # If all lookups fail, try semantic search as last resort
                        logger.warning(f"All direct lookups failed for {doc_id}, trying semantic search")
                        results = app.search(doc_id, 1)  # Use the ID as a search term
                        if results and len(results) > 0:
                            result = results[0]
                            if isinstance(result, dict) and result.get("id") == doc_id:
                                return [result]
                except Exception as e:
                    logger.warning(f"ID lookup failed: {e}")
                    logger.warning(f"Traceback: {traceback.format_exc()}")
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
            
            # Ensure each result has a text field
            enhanced_results = []
            for result in results:
                if isinstance(result, dict):
                    # If result doesn't have text field, try to get it from cache or backend
                    if "id" in result and ("text" not in result or not result["text"]):
                        doc_id = result["id"]
                        # Try cache first
                        text = get_document_from_cache(doc_id)
                        if text:
                            result["text"] = text
                        # If not in cache, try backend
                        elif hasattr(app.embeddings, "backend"):
                            try:
                                doc = app.embeddings.backend.get(doc_id)
                                if doc:
                                    doc_text = doc.get('text') if isinstance(doc, dict) else str(doc)
                                    result["text"] = doc_text
                                    # Add to cache for next time
                                    add_to_document_cache(doc_id, doc_text)
                            except Exception as e:
                                logger.debug(f"Backend lookup failed for result enhancement: {e}")
                    enhanced_results.append(result)
                else:
                    enhanced_results.append(result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to search: {str(e)}")

    @mcp.tool(
        name="add_document",
        description="Add a document to the search index."
    )
    async def add_document(
        ctx: Context,
        text: str,
        id: Optional[str] = None,
        tags: Optional[str] = None,
        metadata: Optional[str] = None
    ) -> Dict:
        """Add a document to the search index.
        
        Args:
            text: Document text
            id: Optional document ID (will be generated if not provided)
            tags: Optional comma-separated tags
            metadata: Optional JSON metadata
        """
        try:
            # Generate ID if not provided
            if not id:
                id = str(uuid.uuid4())
            
            # Parse tags if provided
            tag_list = None
            if tags:
                tag_list = [tag.strip() for tag in tags.split(",")]
            
            # Parse metadata if provided
            meta_dict = None
            if metadata:
                try:
                    meta_dict = json.loads(metadata)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid metadata JSON: {metadata}")
            
            # Create document object
            document = {
                "id": id,
                "text": text
            }
            
            if tag_list:
                document["tags"] = tag_list
            
            if meta_dict:
                document["metadata"] = meta_dict
            
            # Get txtai app
            app = get_txtai_app()
            
            # Add document to index
            logger.info(f"Adding document: {document}")
            app.add(document)
            
            # Add to document cache
            add_to_document_cache(id, text)
            
            # Save index
            if app.config.get("path"):
                logger.info(f"Saving index to: {app.config.get('path')}")
                app.index()
            
            return {
                "status": "success",
                "id": id
            }
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to add document: {str(e)}")

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
