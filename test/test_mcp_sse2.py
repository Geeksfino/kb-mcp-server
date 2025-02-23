"""
MCP client implementation.
"""
import asyncio
import logging
import sys
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_url: str):
        """Connect to MCP server."""
        try:
            logger.debug(f"Connecting to server URL: {server_url}")
            
            # Create SSE transport
            logger.debug("Creating SSE transport...")
            sse_transport = await self.exit_stack.enter_async_context(
                sse_client(server_url)
            )
            self.read_stream, self.write_stream = sse_transport
            
            # Create and initialize session
            logger.debug("Creating client session...")
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.read_stream, self.write_stream)
            )
            
            # Initialize with extended timeout
            logger.debug("Initializing session...")
            await self.session.initialize()
            
            # List available tools
            logger.debug("Requesting tool list...")
            response = await self.session.list_tools()
            tools = response.tools
            print("\nConnected to server with tools:", [tool.name for tool in tools])

            # First add some test content
            print("\nAdding test content...")
            result = await self.session.call_tool(
                "add_content",
                {"content": "This is a test document about artificial intelligence.", "id": "doc1"}
            )
            print(f"Add content result: {result}")

            # Try different search queries
            queries = [
                "AI and machine learning",  # Should find our test doc
                "Who was Super Bowl MVP?",   # Should find SQuAD sports content
                "What is the history of Notre Dame?",  # Should find SQuAD history content
                "Tell me about science and technology"  # Should find both test doc and SQuAD
            ]
            
            for query in queries:
                print(f"\nTesting search with query: {query}")
                result = await self.session.call_tool(
                    "semantic_search",
                    {"query": query, "limit": 5}
                )
                print(f"Search result: {result}")

        except Exception as e:
            logger.error(f"Error connecting to server: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def close(self):
        """Close the client connection."""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_mcp_sse2.py <server_url>")
        sys.exit(1)
        
    server_url = sys.argv[1]
    client = MCPClient()
    
    try:
        await client.connect_to_server(server_url)
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())