import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import sys
import os
import logging

from mcp import ClientSession
from mcp.client.sse import sse_client

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # Ensure logs go to stderr
)

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_url: str):
        """Connect to an MCP server using SSE transport
        
        Args:
            server_url: URL of the MCP server's SSE endpoint
        """
        logger.debug(f"Connecting to server URL: {server_url}")
        
        # Use a longer timeout for server startup
        transport = await self.exit_stack.enter_async_context(
            sse_client(
                server_url,
                timeout=30,  # HTTP operation timeout
                sse_read_timeout=300  # Wait up to 5 minutes for server startup
            )
        )
        self.stdio, self.write = transport
        logger.debug("SSE transport created")
        
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        logger.debug("Client session created")
        
        logger.debug("Starting session initialization...")
        await self.session.initialize()
        logger.debug("Session initialization complete")
        
        # List available tools
        logger.debug("Requesting tool list...")
        response = await self.session.list_tools()
        logger.debug(f"List tools response: {response}")
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

        # First add some test content
        print("\nAdding test content...")
        result = await self.session.call_tool(
            "add_content",
            {"content": "This is a test document about artificial intelligence.", "id": "doc1"}
        )
        print(f"Add content result: {result}")

        # Now try searching
        print("\nTesting semantic search...")
        result = await self.session.call_tool(
            "semantic_search",
            {"query": "AI and machine learning", "limit": 5}
        )
        print(f"Search result: {result}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_mcp_sse.py <server_url>")
        print("Example: python test_mcp_sse.py http://localhost:8000/sse")  # Connect to /sse for SSE events, /messages/ is for POST requests
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())