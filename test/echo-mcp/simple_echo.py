"""
Echo Server using MCP
"""
import json
import logging
import sys
import os
from typing import Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

# Configure logging to both file and stderr
log_dir = os.path.expanduser("~/.codeium/logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "echo_server.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

async def serve() -> None:
    """Run the echo server"""
    logger.debug("Creating MCP server...")
    server = Server("echo-server")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available echo tools."""
        return [
            Tool(
                name="echo",
                description="Echo back the input text",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to echo back",
                        }
                    },
                    "required": ["text"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        """Handle tool calls for echo."""
        try:
            match name:
                case "echo":
                    text = arguments.get("text")
                    if not text:
                        raise ValueError("Missing required argument: text")
                    logger.debug(f"Received echo request with text: {text}")
                    return [TextContent(type="text", text=text)]
                case _:
                    raise ValueError(f"Unknown tool: {name}")
        except Exception as e:
            raise ValueError(f"Error processing echo request: {str(e)}")

    logger.info("Starting Echo Server...")
    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)

def main():
    """MCP Echo Server - Simple echo functionality for MCP"""
    import asyncio
    asyncio.run(serve())

if __name__ == "__main__":
    main()