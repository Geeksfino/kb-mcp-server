#!/usr/bin/env python
"""
MCP client for the txtai MCP server.

This script connects to a running MCP server and queries it with a question.
It prints the answer to stdout.

Usage:
    python qa.py http://localhost:8000/sse "What is the capital of France?"
    python qa.py server.py "What is the capital of France?"
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

class MCPClient:
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
            logger.info(f"Available tools: {[tool.name for tool in response.tools]}")
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to MCP server: {e}", exc_info=True)
            return False
    
    async def ask_question(self, question):
        """Ask a question and return the answer."""
        if not self.session:
            return "Error: Not connected to MCP server"
        
        try:
            # Call the answer_question tool
            result = await self.session.call_tool(
                "answer_question",
                {
                    "question": question
                }
            )
            
            # Extract answer from result
            answer = "No answer found"
            for content in result.content:
                if content.type == "text":
                    answer = content.text
                    break
            
            return answer
        except Exception as e:
            logger.error(f"Error asking question: {e}", exc_info=True)
            return f"Error: {str(e)}"
    
    async def close(self):
        """Close the client connection."""
        if self.exit_stack:
            await self.exit_stack.aclose()

async def main_async():
    """Async main function."""
    # Check if a target and question were provided
    if len(sys.argv) < 3:
        print("Usage:")
        print("  For SSE:   python qa.py http://localhost:8000/sse \"What is your question?\"")
        print("  For stdio: python qa.py server.py \"What is your question?\"")
        return 1
    
    # Get the target and question from command line arguments
    target = sys.argv[1]
    question = " ".join(sys.argv[2:])
    
    # Create client
    client = MCPClient()
    
    try:
        # Connect to the MCP server
        logger.info(f"Connecting to MCP server at {target}")
        
        if not await client.connect(target):
            print(f"Failed to connect to MCP server at {target}")
            return 1
        
        # Ask the question
        logger.info(f"Asking question: {question}")
        answer = await client.ask_question(question)
        
        # Print the result
        print(f"Q: {question}")
        print(f"A: {answer}")
        
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
