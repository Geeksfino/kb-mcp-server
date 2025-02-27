import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import sys
import os
import logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        logger.debug(f"Connecting to server script: {server_script_path}")
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        logger.debug(f"Server parameters: {server_params}")
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        logger.debug("Stdio transport created")
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

        # Use self.session and correct parameter name
        result = await self.session.call_tool("echo", arguments={"text": "hello world!"})
        print("Echo result:", result)
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())