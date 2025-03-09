"""TxtAI MCP Server with causal boost and multilingual support."""
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

# Global configuration variables for causal boost
_ENABLE_CAUSAL_BOOST = False
_CAUSAL_CONFIG_PATH = None

@asynccontextmanager
async def server_lifespan(_: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage txtai application lifecycle.
    
    Uses global variables _ENABLE_CAUSAL_BOOST and _CAUSAL_CONFIG_PATH for configuration.
    """
    global _ENABLE_CAUSAL_BOOST, _CAUSAL_CONFIG_PATH
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
        if _ENABLE_CAUSAL_BOOST:
            try:
                if _CAUSAL_CONFIG_PATH:
                    logger.info(f"Loading custom causal boost configuration from {_CAUSAL_CONFIG_PATH}")
                    causal_config = CausalBoostConfig.load_from_file(_CAUSAL_CONFIG_PATH)
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

def create_server(
    host: str = "localhost",
    port: int = 8000,
    enable_causal_boost: bool = False,
    causal_config_path: Optional[str] = None
) -> FastMCP:
    """Create and configure the MCP server instance.
    
    Args:
        host: Host to bind to when using SSE transport
        port: Port to bind to when using SSE transport
        enable_causal_boost: Whether to enable causal boost feature
        causal_config_path: Path to custom causal boost configuration YAML file
    
    Returns:
        Configured FastMCP server instance
    """
    global _ENABLE_CAUSAL_BOOST, _CAUSAL_CONFIG_PATH
    
    # Configure causal boost
    _ENABLE_CAUSAL_BOOST = enable_causal_boost
    _CAUSAL_CONFIG_PATH = causal_config_path
    
    # Create server with configuration
    server = FastMCP(
        "Knowledgebase Server",
        lifespan=server_lifespan,
        host=host,
        port=port
    )
    
    # Register tools
    register_search_tools(server)
    register_qa_tools(server)
    register_retrieve_tools(server)
    logger.info("Created and configured Knowledgebase instance")
    
    return server

# Create module-level server instance only if not running as main
if __name__ != "__main__":
    mcp = create_server()

# Handle shutdown gracefully
def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

def run():
    import argparse
    
    parser = argparse.ArgumentParser(description='TxtAI MCP Server')
    parser.add_argument('--transport', type=str, default='stdio', choices=['sse', 'stdio'],
                        help='Transport to use (default: stdio)')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host to bind to when using SSE transport (default: localhost)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to bind to when using SSE transport (default: 8000)')
    parser.add_argument('--embeddings', type=str,
                        help='Path to embeddings index')
    
    # Add causal boost arguments
    causal_group = parser.add_argument_group('Causal Boost Configuration')
    causal_group.add_argument('--enable-causal-boost', action='store_true',
                        help='Enable causal boost feature for enhanced relevance scoring')
    causal_group.add_argument('--causal-config', type=str,
                        help='Path to custom causal boost configuration YAML file')
    
    args = parser.parse_args()
    
    # Set environment variable if embeddings path is provided
    if args.embeddings:
        os.environ["TXTAI_EMBEDDINGS"] = args.embeddings
    
    # Configure the server based on arguments
    if args.transport == 'sse':
        os.environ["MCP_SSE_HOST"] = args.host
        os.environ["MCP_SSE_PORT"] = str(args.port)
        logger.info(f"Server will be available at http://{args.host}:{args.port}/sse")
    
        # Create server instance with arguments
        server = create_server(
            host=args.host,
            port=args.port,
            enable_causal_boost=args.enable_causal_boost,
            causal_config_path=args.causal_config
        )
        
        # Let FastMCP handle transport automatically
        # This ensures consistent behavior for causal boost features
        # across direct execution and MCP environments
        server.run(transport=args.transport)
    else:
        # For stdio transport, use default FastMCP run
        logger.info("Server will be available at stdin/stdout")
        server = create_server(
            enable_causal_boost=args.enable_causal_boost,
            causal_config_path=args.causal_config
        )
        server.run()

if __name__ == "__main__":
    run()