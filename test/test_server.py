"""
Test script for the txtai MCP server.
"""
import asyncio
import json
from typing import Dict, Any

from mcp.server.stdio import stdio_server
from txtai_mcp_server.core import create_server


async def send_request(writer: asyncio.StreamWriter, request: Dict[str, Any]) -> None:
    """Send a request to the server."""
    writer.write(json.dumps(request).encode() + b"\n")
    await writer.drain()


async def read_response(reader: asyncio.StreamReader) -> Dict[str, Any]:
    """Read response from the server."""
    line = await reader.readline()
    return json.loads(line.decode())


async def test_semantic_search() -> None:
    """Test semantic search functionality."""
    mcp = create_server()
    
    async with stdio_server() as (reader, writer):
        # Initialize server
        await mcp.run_init(reader, writer, mcp.create_initialization_options())
        
        # Add some test content
        # add_request = {
        #     "method": "tool/call",
        #     "params": {
        #         "name": "add_content",
        #         "arguments": {
        #             "content": "Python is a popular programming language.",
        #             "id": "doc1"
        #         }
        #     }
        # }
        # await send_request(writer, add_request)
        # add_response = await read_response(reader)
        # print("Add content response:", add_response)
        
        # Perform semantic search
        search_request = {
            "method": "tool/call",
            "params": {
                "name": "semantic_search",
                "arguments": {
                    "query": "What programming languages are popular?",
                    "limit": 1
                }
            }
        }
        await send_request(writer, search_request)
        search_response = await read_response(reader)
        print("Search response:", search_response)
        
        # # Test prompt
        # prompt_request = {
        #     "method": "prompt/get",
        #     "params": {
        #         "name": "search_results_analysis",
        #         "arguments": {
        #             "results": search_response["result"],
        #             "query": "What programming languages are popular?"
        #         }
        #     }
        # }
        # await send_request(writer, prompt_request)
        # prompt_response = await read_response(reader)
        # print("Prompt response:", prompt_response)


if __name__ == "__main__":
    asyncio.run(test_semantic_search())