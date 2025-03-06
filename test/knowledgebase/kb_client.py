#!/usr/bin/env python
"""
Knowledge Base MCP client for the txtai MCP server.

This script connects to a running MCP server and allows interactive retrieval
from the knowledge base using the retrieve_context tool.

Usage:
    python kb_client.py http://localhost:8000/sse
    python kb_client.py ../txtai_mcp_server/server.py
"""

import sys
import json
import asyncio
import logging
from contextlib import AsyncExitStack
from urllib.parse import urlparse

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeBaseClient:
    """Client for interacting with the Knowledge Base through MCP."""
    
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()

    async def connect(self, target):
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
            
            read_stream, write_stream = transport
            
            # Create and initialize session
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self.session.initialize()
            
            # List available tools
            response = await self.session.list_tools()
            tools = [tool.name for tool in response.tools]
            logger.info(f"Available tools: {tools}")
            
            # Verify that retrieve_context tool is available
            if "retrieve_context" not in tools:
                logger.warning("retrieve_context tool not found in available tools!")
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to MCP server: {e}", exc_info=True)
            return False
    
    async def retrieve(self, query, limit=5, min_similarity=0.3, causal_boost=True):
        """Retrieve information from the knowledge base."""
        if not self.session:
            return "Error: Not connected to MCP server"
        
        try:
            # Call the retrieve_context tool
            result = await self.session.call_tool(
                "retrieve_context",
                {
                    "query": query,
                    "limit": limit,
                    "min_similarity": min_similarity,
                    "causal_boost": causal_boost
                }
            )
            
            # Extract results from the response
            response = "No results found"
            for content in result.content:
                if content.type == "text":
                    # Parse the JSON response
                    try:
                        results = json.loads(content.text)
                        if results:
                            # Format the results in a more readable way
                            formatted_results = []
                            for i, item in enumerate(results, 1):
                                # Format with score as a percentage and text content
                                score_pct = item.get("score", 0) * 100
                                formatted_results.append(f"Result {i} ({score_pct:.1f}%):\n{item.get('text', '')}\n")
                            
                            response = "\n".join(formatted_results)
                        else:
                            response = "No results found"
                    except json.JSONDecodeError:
                        # If not valid JSON, return as is
                        response = content.text
                    break
            
            return response
        except Exception as e:
            logger.error(f"Error retrieving information: {e}", exc_info=True)
            return f"Error: {str(e)}"
    
    async def close(self):
        """Close the client connection."""
        if self.exit_stack:
            await self.exit_stack.aclose()

async def interactive_session(client):
    """Run an interactive knowledge base retrieval session."""
    print("\n=== TxtAI Knowledge Base Interactive Session ===")
    print("Type 'exit', 'quit', or press Ctrl+C to end the session")
    print("Type 'help' for more information")
    print("------------------------------------------------")
    
    try:
        while True:
            # Get query from user
            query = input("\nQuery: ").strip()
            
            # Check for exit commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("Exiting session...")
                break
            
            # Check for help command
            if query.lower() in ['help', 'h', '?']:
                print("\nCommands:")
                print("  exit, quit, q - Exit the session")
                print("  help, h, ?    - Show this help message")
                print("  limit=N       - Set the result limit (e.g., 'limit=3 what is machine learning?')")
                print("\nAsk any question to retrieve information from the knowledge base.")
                continue
            
            # Skip empty queries
            if not query:
                continue
            
            # Check for limit parameter
            limit = 5  # Default limit
            if query.startswith("limit="):
                try:
                    parts = query.split(" ", 1)
                    limit_str = parts[0].split("=")[1]
                    limit = int(limit_str)
                    query = parts[1] if len(parts) > 1 else ""
                    
                    if not query:
                        print("Please provide a query after the limit parameter.")
                        continue
                except (IndexError, ValueError):
                    print("Invalid limit format. Use 'limit=N query'")
                    continue
            
            # Retrieve information
            logger.info(f"Retrieving information for query: {query} (limit={limit})")
            response = await client.retrieve(query, limit=limit)
            
            # Print the response
            print(f"\nResults:\n{response}")
    
    except KeyboardInterrupt:
        print("\nSession terminated by user.")
    except Exception as e:
        logger.error(f"Error in interactive session: {e}", exc_info=True)
        print(f"\nError: {str(e)}")

async def main_async():
    """Async main function."""
    # Check if a target was provided
    if len(sys.argv) < 2:
        print("Usage:")
        print("  For SSE:   python kb_client.py http://localhost:8000/sse")
        print("  For stdio: python kb_client.py ../txtai_mcp_server/server.py")
        return 1
    
    # Get the target from command line arguments
    target = sys.argv[1]
    
    # Create client
    client = KnowledgeBaseClient()
    
    try:
        # Connect to the MCP server
        logger.info(f"Connecting to MCP server at {target}")
        
        if not await client.connect(target):
            print(f"Failed to connect to MCP server at {target}")
            return 1
        
        # Start interactive session
        await interactive_session(client)
        
        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    finally:
        await client.close()

def main():
    """Main function."""
    return asyncio.run(main_async())

if __name__ == "__main__":
    sys.exit(main())
