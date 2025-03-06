"""TxtAI MCP server implementation."""
import sys
import signal
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, Optional

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions
import mcp.server.stdio as mcp_stdio
import mcp.server.sse as mcp_sse
from txtai.app import Application

from txtai_mcp_server.core.config import TxtAISettings
from txtai_mcp_server.core.context import TxtAIContext
from txtai_mcp_server.core.state import set_txtai_app, get_txtai_app
from txtai_mcp_server.tools.search import register_search_tools
from txtai_mcp_server.tools.qa import register_qa_tools
from txtai_mcp_server.tools.retrieve import register_retrieve_tools

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

# Enable logging for all packages
logging.getLogger('mcp').setLevel(logging.DEBUG)
logging.getLogger('mcp.server.lowlevel.server').setLevel(logging.DEBUG)
logging.getLogger('mcp.server.lowlevel.transport').setLevel(logging.DEBUG)
logging.getLogger('mcp.server.sse').setLevel(logging.DEBUG)
logging.getLogger('mcp.server.stdio').setLevel(logging.DEBUG)
logging.getLogger('txtai').setLevel(logging.DEBUG)
logging.getLogger('txtai_mcp_server').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def txtai_lifespan(mcp: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage txtai application lifecycle."""
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
        
        # Yield serializable context
        yield {"status": "ready"}
        logger.info("Server is ready")
    except Exception as e:
        logger.error(f"Error during lifespan: {e}", exc_info=True)
        raise
    finally:
        logger.info("=== Shutting down txtai server (lifespan) ===")

# Create a server object at the module level for the MCP CLI to use
server = FastMCP("TxtAI Server", lifespan=txtai_lifespan)

# Register tools with the server
register_search_tools(server)
register_qa_tools(server)
register_retrieve_tools(server)
logger.info("Registered search, QA, and retrieve tools with module-level server")

async def run_server(transport: str = 'sse', host: str = 'localhost', port: int = 8000):
    """Run the TxtAI MCP server with the specified transport.
    
    Args:
        transport: The transport to use. Either 'sse' or 'stdio'.
        host: Host to bind to when using SSE transport (default: localhost).
        port: Port to bind to when using SSE transport (default: 8000).
    """
    # Create the server with lifespan
    logger.info("Creating FastMCP instance...")
    mcp = FastMCP(
        "TxtAI Server",
        lifespan=txtai_lifespan,
        host=host,
        port=port
    )
    logger.info("Created FastMCP instance")

    # Register tools
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

    logger.info(f"=== TxtAI server ready with {transport} transport ===")
    if transport == 'sse':
        logger.info(f"Server will be available at http://{host}:{port}/sse")

    # Use the high-level run method which handles the transport internally
    if transport == 'sse':
        # We need to use run_sse_async directly since we're already in an async context
        await mcp.run_sse_async()
    else:
        # We need to use run_stdio_async directly since we're already in an async context
        await mcp.run_stdio_async()

if __name__ == "__main__":
    import argparse
    import os
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='TxtAI MCP Server')
    parser.add_argument('--embeddings', type=str, help='Path to embeddings directory or archive file')
    parser.add_argument('--transport', type=str, default='stdio', choices=['sse', 'stdio'], 
                        help='Transport to use for MCP server (default: stdio)')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host to bind to when using SSE transport (default: localhost)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to bind to when using SSE transport (default: 8000)')
    
    args = parser.parse_args()
    
    # Set environment variable if embeddings path is provided
    if args.embeddings:
        os.environ["TXTAI_EMBEDDINGS"] = args.embeddings
    
    # Run the server with the specified transport
    asyncio.run(run_server(transport=args.transport, host=args.host, port=args.port))

# This function is called by the MCP CLI
def run():
    """Entry point for the MCP CLI."""
    # The MCP CLI will set the transport, so we don't need to specify it here
    asyncio.run(run_server())
