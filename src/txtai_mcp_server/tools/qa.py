"""
Question-answering tools for the txtai MCP server.
"""
import logging
from typing import Dict, List, Optional, Union
import textwrap

from mcp.server.fastmcp import FastMCP, Context
from txtai_mcp_server.core.context import TxtAIContext

logger = logging.getLogger(__name__)

def register_qa_tools(mcp: FastMCP) -> None:
    """Register question-answering tools."""
    logger.debug("Registering QA tools...")
    
    @mcp.tool(
        name="answer_question",
        description="""Extract precise answers to questions using advanced language models.
        Best used for:
        - Getting direct answers to specific questions
        - Extracting precise information from text
        - Verifying facts from content
        - Understanding complex information
        
        Uses hybrid search to find relevant context, combining semantic understanding with keyword matching.
        
        Example: "Who created Python?" -> Will extract "Guido van Rossum" from relevant content."""
    )
    async def answer_question(
        ctx: Context,
        question: str,
        context: Optional[str] = None,
        search_limit: Optional[int] = 3
    ) -> Dict[str, Union[str, float]]:
        """Implementation of question answering using txtai application.
        
        Args:
            question: Question to answer
            context: Optional text to extract from
            search_limit: Number of search results to use if no context
            
        Returns:
            Dict with answer, confidence score, and sources
        """
        logger.debug(f"Answering question: '{question}'")
        txtai_context = get_txtai_context(ctx)
        
        try:
            # Get context from search if not provided
            if not context:
                logger.debug("No context provided, searching...")
                # Use Application's search with SQL query
                sql = """
                    SELECT text, answer, score, source 
                    FROM txtai 
                    WHERE similar(?) 
                    ORDER BY score DESC 
                    LIMIT ?
                """
                results = txtai_context.app.search(sql, (question, search_limit))
                
                if not results:
                    return {
                        "answer": "I could not find any relevant information to answer your question.",
                        "score": 0.0,
                        "sources": []
                    }
                
                # Build context with clear separators and metadata
                contexts = []
                sources = []
                for i, r in enumerate(results, 1):
                    text = r.get("answer") or r.get("text", "")
                    if text:
                        contexts.append(f"Source {i}:\n{textwrap.indent(text, '  ')}")
                        sources.append({
                            "text": r.get("text", "Unknown"),
                            "score": r.get("score", 0.0),
                            "source": r.get("source", "unknown")
                        })
                
                if not contexts:
                    return {
                        "answer": "Found results but could not extract usable context.",
                        "score": 0.0,
                        "sources": []
                    }
                    
                context = "\n\n".join(contexts)
            else:
                sources = [{"text": "Provided context", "score": 1.0, "source": "user"}]
            
            # Use Application's question-answering
            logger.debug("Extracting answer...")
            result = txtai_context.app.extract([(question, context)])[0]
            
            answer = {
                "answer": result["answer"],
                "score": float(result["score"]),
                "sources": sources
            }
            
            logger.info(f"Answer: {answer['answer']}")
            logger.info(f"Confidence: {answer['score']:.3f}")
            
            return answer
            
        except Exception as e:
            logger.error(f"QA failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to answer question: {str(e)}")
            
    logger.debug("QA tools registered successfully")
