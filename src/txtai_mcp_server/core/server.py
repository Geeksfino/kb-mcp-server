"""
Core MCP server implementation for txtai.
"""
import logging
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any

from mcp.server.fastmcp import FastMCP, Context
from txtai.embeddings import Embeddings

from .context import TxtAIContext
from ..tools import register_search_tools
from ..resources import register_config_resources, register_model_resources
from ..prompts import register_search_prompts


logger = logging.getLogger(__name__)


@asynccontextmanager
async def txtai_lifespan(ctx: Context) -> AsyncIterator[Dict[str, Any]]:
    """Initialize txtai components on startup."""
    print("DEBUG: Entering txtai_lifespan")  # Print directly to ensure it shows up
    logger.critical("Entering txtai_lifespan")  # Log at CRITICAL level
    try:
        logger.info("Loading txtai components...")
        t0 = time.time()
        
        # Initialize embeddings with a default model
        logger.info("Initializing embeddings model...")
        embeddings_config = {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "method": "transformers",  # Use transformers as the method
            "transform": "mean",  # Use mean pooling for sentence embeddings
            "normalize": True,  # Normalize vectors
            "gpu": False  # Don't use GPU for now
        }
        embeddings = Embeddings(embeddings_config)
        
        # Load embeddings model
        t1 = time.time()
        logger.info("Loading embeddings model...")
        # Create an empty index first
        embeddings.index([])
        logger.info(f"Embeddings model loaded in {time.time() - t1:.2f}s")
        
        context = TxtAIContext(embeddings=embeddings)
        lifespan_context = {"txtai_context": context}
        logger.info(f"Total txtai initialization time: {time.time() - t0:.2f}s")
        yield lifespan_context
    except Exception as e:
        logger.error("Error in txtai_lifespan: %s", e)
        logger.error("Traceback: %s", traceback.format_exc())
        raise
    finally:
        try:
            # Cleanup if needed - txtai components generally don't need explicit cleanup
            print("DEBUG: Exiting txtai_lifespan")  # Print directly to ensure it shows up
            logger.critical("Exiting txtai_lifespan")  # Log at CRITICAL level
        except Exception as e:
            logger.error("Error in txtai_lifespan cleanup: %s", e)
            logger.error("Traceback: %s", traceback.format_exc())


def create_server() -> FastMCP:
    """Create and configure the MCP server."""
    try:
        # Create server instance
        mcp = FastMCP(
            "TxtAI Server",
            lifespan=txtai_lifespan,
            dependencies=["txtai", "torch", "transformers"]
        )
        
        # Register tools
        register_search_tools(mcp)
        
        # Register resources
        register_config_resources(mcp)
        register_model_resources(mcp)
        
        # Register prompts
        register_search_prompts(mcp)
        
        return mcp
    except Exception as e:
        logger.error("Error creating server: %s", e)
        logger.error("Traceback: %s", traceback.format_exc())
        raise
