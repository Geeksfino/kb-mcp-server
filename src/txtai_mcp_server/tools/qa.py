"""Question-answering tools for the txtai MCP server."""
import logging
import sys
import traceback
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import Field

from ..core.context import TxtAIContext
from .search import get_txtai_context

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
        """Answer questions using txtai."""
        logger.debug(f"QA request - question: {question}, limit: {limit}")
        try:
            txtai_ctx = get_txtai_context(ctx)
            answer = txtai_ctx.app.question(question, limit=limit)
            return answer
        except Exception as e:
            logger.error(f"Error in question answering: {str(e)}\n{traceback.format_exc()}")
            raise
