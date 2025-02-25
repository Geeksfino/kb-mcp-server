"""TxtAI MCP server implementation."""
import sys
import signal
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any

from mcp.server.fastmcp import FastMCP, Context
from txtai.app import Application

from txtai_mcp_server.core.config import TxtAISettings
from txtai_mcp_server.core.context import TxtAIContext
from txtai_mcp_server.tools.search import register_search_tools
from txtai_mcp_server.tools.qa import register_qa_tools

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
logging.getLogger('mcp.server.sse').setLevel(logging.DEBUG)  # Add SSE logging
logging.getLogger('mcp.server.stdio').setLevel(logging.DEBUG)  # Add stdio transport logging
logging.getLogger('txtai').setLevel(logging.DEBUG)
logging.getLogger('txtai_mcp_server').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def server_lifespan(_: Context) -> AsyncIterator[Dict[str, Any]]:
    """Server lifespan."""
    logger.info("=== Starting txtai server (lifespan) ===")
    try:
        yield {"status": "ready"}
        logger.info("Server is ready")
    finally:
        logger.info("=== Shutting down txtai server (lifespan) ===")

# Create the server
logger.info("Creating FastMCP instance...")
mcp = FastMCP(
    "TxtAI Server",
    lifespan=server_lifespan
)
logger.info("Created FastMCP instance")

# Register tools
register_search_tools(mcp)
register_qa_tools(mcp)
logger.info("Registered search and QA tools")

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
