"""
Model-related resources for the txtai MCP server.
"""
import json
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP
from transformers import AutoConfig

from ..core import TxtAIContext


def register_model_resources(mcp: FastMCP) -> None:
    """Register model-related resources with the MCP server."""
    
    @mcp.resource("model://embeddings/{name}")
    def model_info(name: str) -> str:
        """
        Get information about a specific model.
        
        Args:
            name: Model name or path
            
        Returns:
            JSON string of model information
        """
        try:
            # Get model config from HuggingFace
            config = AutoConfig.from_pretrained(name)
            
            info = {
                "name": name,
                "architecture": config.architectures[0] if config.architectures else None,
                "hidden_size": config.hidden_size,
                "vocab_size": config.vocab_size,
                "model_type": config.model_type
            }
            return json.dumps(info, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2)
    
    @mcp.resource("model://pipeline/{name}")
    def pipeline_info(name: str) -> str:
        """
        Get information about a specific pipeline.
        
        Args:
            name: Pipeline name
            
        Returns:
            JSON string of pipeline information
        """
        ctx: TxtAIContext = mcp.request_context.lifespan_context
        if not ctx.pipelines or name not in ctx.pipelines:
            raise RuntimeError(f"Pipeline {name} not initialized")
            
        pipeline = ctx.pipelines[name]
        info = {
            "name": name,
            "type": pipeline.__class__.__name__,
            "model": getattr(pipeline, "model", None),
            "task": getattr(pipeline, "task", None),
            "methods": [
                method for method in dir(pipeline) 
                if not method.startswith("_") and callable(getattr(pipeline, method))
            ]
        }
        return json.dumps(info, indent=2)
    
    @mcp.resource("model://capabilities")
    def model_capabilities() -> str:
        """
        Get information about all available models and capabilities.
        
        Returns:
            JSON string of capabilities information
        """
        ctx: TxtAIContext = mcp.request_context.lifespan_context
        
        capabilities = {
            "embeddings": {
                "model": ctx.embeddings.path if ctx.embeddings else None,
                "dimension": ctx.embeddings.dimension if ctx.embeddings else None,
                "operations": ["search", "add", "delete", "similarity"]
            },
            "pipelines": {
                name: {
                    "type": pipeline.__class__.__name__,
                    "operations": [
                        method for method in dir(pipeline) 
                        if not method.startswith("_") and callable(getattr(pipeline, method))
                    ]
                }
                for name, pipeline in (ctx.pipelines or {}).items()
            }
        }
        return json.dumps(capabilities, indent=2)
