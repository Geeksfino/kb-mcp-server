"""
Text processing tools for the txtai MCP server.
"""
from typing import List, Dict, Any, Optional

from mcp.server.fastmcp import FastMCP

from ..core import TxtAIContext


def register_text_tools(mcp: FastMCP) -> None:
    """Register text processing tools with the MCP server."""
    
    @mcp.tool()
    def extract_text(content: str, format: Optional[str] = None) -> str:
        """
        Extract text from various content formats.
        
        Args:
            content: Content to extract text from
            format: Optional format hint (e.g., 'html', 'pdf')
            
        Returns:
            Extracted text
        """
        ctx: TxtAIContext = mcp.request_context.lifespan_context["txtai_context"]
        if not ctx.pipelines or "textractor" not in ctx.pipelines:
            raise RuntimeError("Textractor pipeline not initialized")
            
        # Extract text
        extracted = ctx.pipelines["textractor"](content)
        return extracted
