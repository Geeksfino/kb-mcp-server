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
import json
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

            # Test documents
            test_documents = [
                "US tops 5 million confirmed virus cases",
                "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
                "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
                "The National Park Service warns against sacrificing slower friends in a bear attack",
                "Maine man wins $1M from $25 lottery ticket",
                "Make huge profits without work, earn up to $100,000 a day"
            ]
            
            # Add test documents to the index
            print("\nAdding test documents...")
            for i, text in enumerate(test_documents):
                await self.session.call_tool(
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
                "public health story",
                "war",
                "wildlife",
                "asia",
                "lucky",
                "dishonest junk"
            ]
            
            # Print header for the table
            print("\nQuery                Best Match")
            print("--------------------------------------------------")
            
            # Reduce logging level temporarily to make output cleaner
            original_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.ERROR)
            
            # Process each query and display results in a table format
            for query in queries:
                result = await self.session.call_tool(
                    "semantic_search",
                    {"query": query, "limit": 1}
                )
                
                # Parse the JSON result
                content_text = str(result.content)
                
                # Default text if no results
                result_text = "No results found"
                
                # Extract the document ID from the result
                doc_id = None
                
                # Handle TextContent wrapper if present
                if "TextContent" in content_text:
                    import re
                    json_match = re.search(r"text='(\[.*?\])'", content_text)
                    if json_match:
                        json_str = json_match.group(1)
                        try:
                            results = json.loads(json_str)
                            if results and len(results) > 0:
                                doc_id = results[0].get("id")
                        except json.JSONDecodeError:
                            pass
                # Regular JSON parsing as fallback
                elif content_text and "[" in content_text and "]" in content_text:
                    try:
                        results = json.loads(content_text)
                        if results and len(results) > 0:
                            doc_id = results[0].get("id")
                    except json.JSONDecodeError:
                        pass
                
                # Look up the document text using the ID
                if doc_id and doc_id.startswith("doc"):
                    try:
                        doc_num = int(doc_id[3:])
                        if 1 <= doc_num <= len(test_documents):
                            result_text = test_documents[doc_num - 1]
                    except (ValueError, IndexError):
                        pass
                
                # Format and print the result in a table row
                print(f"%-20s %s" % (query, result_text))
            
            # Restore original logging level
            logging.getLogger().setLevel(original_level)

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
