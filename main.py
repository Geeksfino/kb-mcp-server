"""Simple echo server for testing MCP."""
import sys
import signal
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, Optional

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.lowlevel.server import Server

from txtai_mcp_server.core.config import TxtAISettings
from txtai_mcp_server.core.context import TxtAIContext
from txtai_mcp_server.core.state import set_txtai_app, get_txtai_app, set_causal_config
from txtai_mcp_server.tools.causal_config import CausalBoostConfig, DEFAULT_CAUSAL_CONFIG
from txtai_mcp_server.tools.search import register_search_tools
from txtai_mcp_server.tools.qa import register_qa_tools
from txtai_mcp_server.tools.retrieve import register_retrieve_tools


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

# Enable MCP logging
logging.getLogger('mcp').setLevel(logging.DEBUG)
logging.getLogger('mcp.server.lowlevel.server').setLevel(logging.DEBUG)
logging.getLogger('mcp.server.lowlevel.transport').setLevel(logging.DEBUG)
logging.getLogger('mcp.server.sse').setLevel(logging.DEBUG)  # Add SSE logging
logging.getLogger('mcp.server.stdio').setLevel(logging.DEBUG)  # Add stdio transport logging
logger = logging.getLogger(__name__)

@asynccontextmanager
async def server_lifespan(_: FastMCP, enable_causal_boost: bool = False, causal_config_path: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
    """Manage txtai application lifecycle.
    
    Args:
        mcp: FastMCP server instance
        enable_causal_boost: Whether to enable causal boost feature
        causal_config_path: Path to custom causal boost configuration file
    """
    logger.info("=== Starting txtai server (lifespan) ===")
    try:
        # Initialize application
        if os.environ.get("TXTAI_EMBEDDINGS"):
            # Environment variable has priority
            embeddings_path = os.environ.get("TXTAI_EMBEDDINGS")
            logger.info(f"Loading embeddings from environment variable TXTAI_EMBEDDINGS: {embeddings_path}")
            settings, app = TxtAISettings.from_embeddings(embeddings_path)
            logger.debug(f"Loaded TxtAI settings from embeddings: {settings.dict()}")
        else:
            # Load from environment variables or config file
            settings = TxtAISettings.load()
            logger.debug(f"Loaded TxtAI settings: {settings.dict()}")
            app = settings.create_application()
            logger.debug(f"Created txtai application with configuration:")
            logger.debug(f"- Model path: {app.config.get('path')}")
            logger.debug(f"- Content storage: {app.config.get('content')}")
            logger.debug(f"- Embeddings config: {app.config.get('embeddings')}")
            logger.debug(f"- Extractor config: {app.config.get('extractor')}")
        
        set_txtai_app(app)
        logger.info("Created txtai application")
        
        # Initialize causal boost configuration if enabled
        if enable_causal_boost:
            try:
                if causal_config_path:
                    logger.info(f"Loading custom causal boost configuration from {causal_config_path}")
                    causal_config = CausalBoostConfig.load_from_file(causal_config_path)
                else:
                    logger.info("Using default causal boost configuration")
                    causal_config = DEFAULT_CAUSAL_CONFIG
                set_causal_config(causal_config)
                logger.info("Initialized causal boost configuration")
            except Exception as e:
                logger.error(f"Failed to initialize causal boost configuration: {e}")
                raise
        
        # Yield serializable context
        yield {"status": "ready"}
        logger.info("Server is ready")
    except Exception as e:
        logger.error(f"Error during lifespan: {e}", exc_info=True)
        raise
    finally:
        logger.info("=== Shutting down txtai server (lifespan) ===")


# Create the server
logger.info("Creating Knowledgebase instance...")
mcp = FastMCP(
    "Knowledgebase Server",
    lifespan=server_lifespan
)
logger.info("Created Knowledgebase instance")
# Register tools with the server
register_search_tools(mcp)
register_qa_tools(mcp)
register_retrieve_tools(mcp)
logger.info("Registered search, QA, and retrieve tools with module-level server")

# Handle shutdown gracefully
def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

logger.info("=== TxtAI server ready ===")

if __name__ == "__main__":
    mcp.run()