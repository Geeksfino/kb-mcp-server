"""
Entry point for the txtai MCP server.
"""
import logging
import sys
import traceback

from txtai_mcp_server.core import create_server


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Ensure logs go to stderr
)

# Enable logging for all packages
logging.getLogger("txtai_mcp_server").setLevel(logging.DEBUG)
logging.getLogger("txtai").setLevel(logging.DEBUG)
logging.getLogger("mcp").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    # Create server instance at module level for mcp run
    logger.info("Creating TxtAI MCP server...")
    mcp = create_server()
    logger.info("Server created with name: %s", mcp.name)
except Exception as e:
    logger.error("Failed to create server: %s", e)
    logger.error("Traceback: %s", traceback.format_exc())
    sys.exit(1)

if __name__ == "__main__":
    mcp.run()