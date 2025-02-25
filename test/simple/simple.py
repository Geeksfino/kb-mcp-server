"""
Simple test script for txtai MCP server.
Supports both stdio and SSE transports.

Usage:
    # For stdio transport (when running mcp run main.py):
    python test/simple/simple.py main.py
    
    # For SSE transport (when running server separately):
    python test/simple/simple.py http://localhost:8000/sse
"""
import asyncio
import logging
import sys
from contextlib import AsyncExitStack
from typing import Optional
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect(self, target: str):
        """Connect to MCP server using appropriate transport.
        
        Args:
            target: Either a URL (for SSE) or script path (for stdio)
        """
        try:
            # Determine transport type
            is_url = urlparse(target).scheme != ''
            
            # Create appropriate transport
            if is_url:
                logger.info(f"Using SSE transport with URL: {target}")
                transport = await self.exit_stack.enter_async_context(
                    sse_client(target)
                )
            else:
                logger.info(f"Using stdio transport with script: {target}")
                transport = await self.exit_stack.enter_async_context(
                    stdio_client([sys.executable, target])
                )
            
            self.read_stream, self.write_stream = transport
            
            # Create and initialize session
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.read_stream, self.write_stream)
            )
            await self.session.initialize()
            
            # List available tools
            response = await self.session.list_tools()
            print("\nAvailable tools:", [tool.name for tool in response.tools])

            # Add test documents from txtai intro
            print("\nAdding test documents...")
            documents = [
                "US tops 5 million confirmed virus cases",
                "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
                "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
                "The National Park Service warns against sacrificing slower friends in a bear attack",
                "Maine man wins $1M from $25 lottery ticket",
                "Make huge profits without work, earn up to $100,000 a day"
            ]
            
            for i, text in enumerate(documents):
                result = await self.session.call_tool(
                    "add_content",
                    {
                        "text": text,
                        "metadata": {"id": f"doc{i+1}"}
                    }
                )
                print(f"Added document {i+1}")

            # Test semantic search
            queries = [
                "feel good story",
                "climate change",
                "wildlife",
                "scam"
            ]
            
            for query in queries:
                print(f"\nSearching for: {query}")
                result = await self.session.call_tool(
                    "semantic_search",
                    {"query": query, "limit": 1}
                )
                print(f"Result: {result[0]['text'] if result else 'No results'}")
                print(f"Score: {result[0]['score'] if result else 'N/A'}")

        except Exception as e:
            logger.error(f"Error during test: {e}")
            raise

    async def close(self):
        """Close the client connection."""
        if self.exit_stack:
            await self.exit_stack.aclose()

async def main():
    """Run the test client."""
    if len(sys.argv) != 2:
        print("Usage:")
        print("  For stdio: python simple.py main.py")
        print("  For SSE:   python simple.py http://localhost:8000/sse")
        sys.exit(1)
        
    target = sys.argv[1]
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    client = MCPClient()
    try:
        await client.connect(target)
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
