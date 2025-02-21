"""
Configuration resources for the txtai MCP server.
"""
import json
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP

from ..core import TxtAIContext


def register_config_resources(mcp: FastMCP) -> None:
    """Register configuration-related resources with the MCP server."""
    
    @mcp.resource("config://embeddings")
    def embeddings_config() -> str:
        """
        Get the current embeddings configuration.
        
        Returns:
            JSON string of embeddings configuration
        """
        ctx: TxtAIContext = mcp.request_context.lifespan_context
        if not ctx.embeddings:
            raise RuntimeError("Embeddings not initialized")
            
        config = {
            "path": ctx.embeddings.path,
            "dimension": ctx.embeddings.dimension,
            "backend": ctx.embeddings.backend.__class__.__name__
        }
        return json.dumps(config, indent=2)
    
    @mcp.resource("config://pipelines")
    def pipeline_config() -> str:
        """
        Get the current pipeline configurations.
        
        Returns:
            JSON string of pipeline configurations
        """
        ctx: TxtAIContext = mcp.request_context.lifespan_context
        if not ctx.pipelines:
            raise RuntimeError("Pipelines not initialized")
            
        config = {}
        for name, pipeline in ctx.pipelines.items():
            config[name] = {
                "type": pipeline.__class__.__name__,
                "model": getattr(pipeline, "model", None),
                "task": getattr(pipeline, "task", None)
            }
        return json.dumps(config, indent=2)
    
    @mcp.resource("config://server")
    def server_config() -> str:
        """
        Get the server configuration.
        
        Returns:
            JSON string of server configuration
        """
        config = {
            "name": mcp.name,
            "version": getattr(mcp, "version", "0.1.0"),
            "dependencies": mcp.dependencies
        }
        return json.dumps(config, indent=2)
