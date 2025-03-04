"""Question-answering tools for the txtai MCP server."""
import logging
import sys
import traceback
import json
from typing import Dict, List, Optional, Any, Union

from mcp.server.fastmcp import FastMCP, Context
from pydantic import Field

from ..core.context import TxtAIContext
from ..core.state import get_txtai_app, get_document_cache

logger = logging.getLogger(__name__)

def register_qa_tools(mcp: FastMCP) -> None:
    """Register QA-related tools with the MCP server."""
    logger.debug("Starting registration of QA tools...")
    
    @mcp.tool(
        name="answer_question",
        description="""Answer questions using AI-powered question answering.
        Best used for:
        - Getting specific answers from documents
        - Finding factual information
        - Extracting precise details
        
        Uses semantic search to find relevant passages and then extracts the answer.
        
        Example: "What is the maximum batch size?" will return the specific batch size value."""
    )
    async def answer_question(
        ctx: Context,
        question: str,
        limit: Optional[int] = Field(3, description="Maximum number of passages to search through"),
    ) -> str:
        """Answer questions using txtai's capabilities."""
        logger.info(f"QA request - question: {question}, limit: {limit}")
        try:
            app = get_txtai_app()
            
            # First approach: Try using extractor pipeline if available
            if hasattr(app, "pipelines") and "extractor" in getattr(app, "pipelines", {}) and hasattr(app, "extract"):
                logger.info("Using extractor pipeline for question answering")
                
                # First search for relevant documents
                search_results = app.search(question, limit=limit)
                
                if not search_results:
                    logger.info("No search results found")
                    return "No relevant information found to answer the question."
                
                # Extract texts from search results
                texts = []
                for result in search_results:
                    if isinstance(result, dict):
                        if "text" in result:
                            texts.append(result["text"])
                        elif "id" in result and get_document_cache().get(result["id"]):
                            texts.append(get_document_cache().get(result["id"]))
                
                if not texts:
                    logger.info("No texts found in search results")
                    return "No text content available to answer the question."
                
                # Create extraction queue
                queue = [(None, question, question, False)]
                
                # Extract answers
                answers = app.extract(queue, texts)
                
                if answers and answers[0] and len(answers[0]) > 1:
                    return answers[0][1]  # Return the answer part
                else:
                    logger.info("No answer extracted, falling back to search")
            
            # Second approach: Use SQL-based search to find similar questions and answers
            logger.info("Using SQL-based search for question answering")
            
            # Escape the question for SQL
            safe_question = escape_sql_string(question)
            
            # Try to get the answer field if it exists
            try:
                sql_query = f"select text, answer, score from txtai where similar('{safe_question}') limit 1"
                results = app.search(sql_query)
                
                if results and len(results) > 0 and "answer" in results[0]:
                    return results[0]["answer"]
            except Exception as e:
                logger.info(f"Error getting answer field: {str(e)}, falling back to text field")
            
            # Third approach: Just return the most similar text
            results = app.search(question, limit=1)
            
            if results and len(results) > 0:
                if isinstance(results[0], dict):
                    return results[0].get("text", "No answer found")
                else:
                    return str(results[0])
            else:
                return "No answer found"
                
        except Exception as e:
            logger.error(f"Error in question answering: {str(e)}\n{traceback.format_exc()}")
            return f"Error processing question: {str(e)}"

def escape_sql_string(text: str) -> str:
    """Escape a string for use in SQL queries."""
    if text is None:
        return text
    return text.replace("'", "''")