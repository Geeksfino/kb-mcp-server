"""Simple echo server for testing MCP."""
import sys
import signal
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.lowlevel.server import Server

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
async def server_lifespan(_: Context) -> AsyncIterator[Dict[str, Any]]:
    """Server lifespan."""
    logger.info("=== Starting echo server (lifespan) ===")
    try:
        yield {"status": "ready"}
        logger.info("Server is ready")
    finally:
        logger.info("=== Shutting down echo server (lifespan) ===")

class LoggingServer(Server):
    """Server with extra logging."""
    async def handle_message(self, message):
        logger.debug(f"Received message: {message}")
        try:
            result = await super().handle_message(message)
            logger.debug(f"Sending response: {result}")
            return result
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            raise

# Create the server
logger.info("Creating FastMCP instance...")
mcp = FastMCP(
    "Echo Server",
    lifespan=server_lifespan,
    server_class=LoggingServer
)
logger.info("Created FastMCP instance")

@mcp.tool()
async def echo(ctx: Context, text: str) -> str:
    """Echo back the message."""
    logger.info(f"Echo called with text: {text}")
    return f"Echo: {text}"

logger.info("Registered echo tool")

# Handle shutdown gracefully
def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

logger.info("=== Echo server ready ===")

if __name__ == "__main__":
    mcp.run()