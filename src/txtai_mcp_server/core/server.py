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
from mcp.server.lowlevel.server import Server as MCPServer
from txtai.embeddings import Embeddings

logger = logging.getLogger(__name__)

class TxtAIServer(MCPServer):
    """Custom MCP server with enhanced logging and context management."""
    
    def __init__(self, name: str, instructions: str | None = None):
        logger.debug("Initializing TxtAIServer...")
        super().__init__(name, instructions)
        self._lifespan_context = None
        logger.debug("TxtAIServer initialized")

    @property
    def lifespan_context(self):
        return self._lifespan_context

    @lifespan_context.setter
    def lifespan_context(self, context):
        logger.debug(f"Setting lifespan context: {context}")
        self._lifespan_context = context

    async def handle_message(self, message):
        """Handle messages with detailed logging."""
        logger.debug(f"Received message: {message}")
        try:
            result = await super().handle_message(message)
            logger.debug(f"Message handled successfully: {result}")
            return result
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

@asynccontextmanager
async def txtai_lifespan(ctx: Context) -> AsyncIterator[Dict[str, Any]]:
    """Manage txtai components lifecycle."""
    logger.info("=== Starting txtai server initialization ===")
    try:
        t0 = time.time()
        
        # Initialize embeddings
        logger.info("Initializing embeddings...")
        embeddings_config = {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "method": "transformers",
            "transform": "mean",
            "normalize": True,
            "content": True,  # Store content in the index
            "gpu": True  # Try GPU first
        }
        
        try:
            embeddings = Embeddings(embeddings_config)
            logger.info("Successfully initialized embeddings with GPU support")
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings with GPU: {e}. Falling back to CPU.")
            embeddings_config["gpu"] = False
            embeddings = Embeddings(embeddings_config)
            logger.info("Successfully initialized embeddings on CPU")
        
        # Create initial index
        logger.info("Creating initial empty index...")
        embeddings.index([])
        
        # Set up context
        from .context import TxtAIContext
        context = TxtAIContext(embeddings=embeddings)
        lifespan_context = {"txtai_context": context}
        
        if hasattr(ctx, 'server'):
            logger.debug("Setting lifespan context on server...")
            ctx.server.lifespan_context = lifespan_context
        
        logger.info(f"Initialization completed in {time.time() - t0:.2f}s")
        yield lifespan_context
        
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        logger.info("=== Shutting down txtai server ===")

def create_server() -> FastMCP:
    """Create and configure the MCP server."""
    logger.info("Creating TxtAI MCP server...")
    try:
        # Create server instance
        mcp = FastMCP(
            "TxtAI Server",
            lifespan=txtai_lifespan,
            server_class=TxtAIServer,
            dependencies=["txtai", "torch", "transformers"]
        )
        
        # Register components
        from ..tools import register_search_tools
        from ..resources import register_config_resources, register_model_resources
        from ..prompts import register_search_prompts
        
        logger.debug("Registering tools...")
        register_search_tools(mcp)
        
        logger.debug("Registering resources...")
        register_config_resources(mcp)
        register_model_resources(mcp)
        
        logger.debug("Registering prompts...")
        register_search_prompts(mcp)
        
        logger.info("Server created with name: TxtAI Server")
        return mcp
        
    except Exception as e:
        logger.error(f"Error creating server: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise