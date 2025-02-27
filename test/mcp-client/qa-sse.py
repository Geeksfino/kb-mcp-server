"""
MCP client implementation for testing SSE transport.
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

            # First add some test QA pairs
            print("\nAdding test content...")
            
            qa_pairs = [
                ("what is the name of justin bieber brother?", "Jazmyn Bieber, Jaxon Bieber"),
                ("what character did natalie portman play in star wars?", "Padmé Amidala"),
                ("what state does selena gomez?", "New York City"),
                ("what country is the grand bahama island in?", "Bahamas"),
                ("what kind of money to take to bahamas?", "Bahamian dollar"),
                ("what time zone is new york under?", "North American Eastern Time Zone")
            ]
            
            # Add each QA pair
            for i, (question, answer) in enumerate(qa_pairs):
                result = await self.session.call_tool(
                    "add_content",
                    {
                        "content": f"{question} {answer}",
                        "id": f"qa{i+1}",
                        "is_question": True
                    }
                )
                print(f"Add QA content result {i+1}: {result}")

            # Test different queries from the example notebook
            queries = [
                "What is the timezone of NYC?",
                "Things to do in New York",
                "What is the timezone of Florida?",
                "Who is Justin Bieber's sibling?",
                "Tell me about Natalie Portman in Star Wars"
            ]
            
            for query in queries:
                print(f"\nTesting search with query: {query}")
                result = await self.session.call_tool(
                    "semantic_search",
                    {"query": query, "limit": 2}  # Limit to top 2 results for clarity
                )
                print(f"Search result: {result}")

        except Exception as e:
            logger.error(f"Error connecting to server: {e}")
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