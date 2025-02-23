"""
Core MCP server implementation for txtai.
"""
import logging
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any
from pathlib import Path

import torch
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.lowlevel.server import Server as MCPServer
from txtai.embeddings import Embeddings

from .config import TxtAISettings
from .context import TxtAIContext

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
        
        # Load settings from .env
        settings = TxtAISettings.load()
        logger.info(f"Storage mode: {settings.storage_mode}")
        
        embeddings_config = settings.get_embeddings_config()
        logger.debug(f"Embeddings config created")
        
        # Initialize embeddings with GPU fallback
        try:
            embeddings = Embeddings(embeddings_config)
            
            if settings.storage_mode == "persistence":
                # Try to load existing embeddings, but don't crash if they don't exist
                storage_path = Path(settings.index_path).expanduser()
                if storage_path.exists():
                    try:
                        logger.info(f"Loading embeddings from {storage_path}")
                        embeddings.load(str(storage_path))
                        logger.info("Successfully loaded existing embeddings")
                    except Exception as e:
                        logger.warning(f"Failed to load embeddings from {storage_path}: {e}")
                        logger.info("Starting with empty embeddings")
                else:
                    logger.info(f"No existing embeddings at {storage_path}, starting fresh")
                    
            elif settings.dataset_enabled and settings.dataset_name:
                # Load from HuggingFace dataset if configured
                try:
                    from datasets import load_dataset
                    logger.info(f"Loading dataset: {settings.dataset_name} (split: {settings.dataset_split})")
                    dataset = load_dataset(settings.dataset_name, split=settings.dataset_split)
                    
                    # Auto-detect text field if not specified
                    text_field = settings.dataset_text_field
                    if not text_field:
                        text_fields = ["text", "content", "article", "document"]
                        for field in text_fields:
                            if field in dataset.features:
                                text_field = field
                                break
                        if not text_field:
                            raise ValueError(f"Could not find text field in dataset. Available fields: {list(dataset.features.keys())}")
                    
                    # Index dataset
                    logger.info(f"Indexing {len(dataset)} documents from field '{text_field}'")
                    embeddings.index([(i, item[text_field], None) for i, item in enumerate(dataset)])
                    logger.info("Successfully loaded and indexed dataset")
                except Exception as e:
                    logger.error(f"Failed to load dataset: {e}")
                    logger.warning("Starting with empty embeddings")
            else:
                logger.info("Starting with empty embeddings (memory mode, no dataset configured)")
            
        except Exception as e:
            if embeddings_config["gpu"]:
                logger.warning(f"Failed to initialize embeddings with GPU: {e}. Falling back to CPU.")
                embeddings_config["gpu"] = False
                embeddings = Embeddings(embeddings_config)
                logger.info("Successfully initialized embeddings on CPU")
            else:
                raise
        
        # Create context
        txtai_context = TxtAIContext(embeddings=embeddings)
        logger.info(f"Server initialization completed in {time.time() - t0:.2f}s")
        
        yield {"txtai_context": txtai_context}
        
        # Cleanup
        if settings.storage_mode == "persistence":
            storage_path = Path(settings.index_path).expanduser()
            try:
                logger.info(f"Saving embeddings to {storage_path}")
                storage_path.parent.mkdir(parents=True, exist_ok=True)
                embeddings.save(str(storage_path))
                logger.info("Successfully saved embeddings")
            except Exception as e:
                logger.error(f"Failed to save embeddings: {e}")
                
    except Exception as e:
        logger.error(f"Error in txtai lifespan: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

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