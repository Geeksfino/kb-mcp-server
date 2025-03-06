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

def extract_text_from_result(json_str, doc_id_to_text=None):
    """Extract text from search result JSON."""
    result_text = "No results found"
    
    try:
        # Handle TextContent wrapper if present
        if "TextContent" in json_str:
            import re
            json_match = re.search(r"text='(\[.*?\])'", json_str)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = json_str.split("text=")[1]
                if json_str.startswith("'") and json_str.endswith("'"):
                    json_str = json_str[1:-1]
        
        # Check for error message
        if "Error executing tool" in json_str:
            print(f"Content text: {json_str}")
            return "No results found"
        
        # Remove any surrounding quotes
        if json_str.startswith("'") and json_str.endswith("'"):
            json_str = json_str[1:-1]
        
        # Parse the JSON
        results = json.loads(json_str)
        
        # Extract text and ID from the first result
        if results and len(results) > 0:
            first_result = results[0]
            if isinstance(first_result, dict):
                # Get document ID
                doc_id = first_result.get("id")
                
                # Get document text
                if "text" in first_result and first_result["text"] != "No text available":
                    result_text = first_result["text"]
                elif doc_id and doc_id_to_text and doc_id in doc_id_to_text:
                    # Fallback to document cache if text not in result
                    result_text = doc_id_to_text[doc_id]
            elif isinstance(first_result, (list, tuple)) and len(first_result) >= 2:
                # Handle (id, score) tuple format
                doc_id = first_result[0]
                if isinstance(doc_id, str) and doc_id_to_text and doc_id in doc_id_to_text:
                    result_text = doc_id_to_text[doc_id]
                elif isinstance(doc_id, int) and doc_id_to_text:
                    # Convert numeric ID to string key format
                    doc_key = f"doc{doc_id+1}"
                    if doc_key in doc_id_to_text:
                        result_text = doc_id_to_text[doc_key]
                
    except Exception as e:
        print(f"Error parsing result: {e}")
        print(f"Content text: {json_str}")
    
    return result_text

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

            # Test documents with specific IDs to match the reference notebook
            test_documents = {
                "doc1": "Maine man wins $1M from $25 lottery ticket",
                "doc2": "Make huge profits without work, earn up to $100,000 a day",
                "doc3": "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
                "doc4": "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
                "doc5": "The National Park Service warns against sacrificing slower friends in a bear attack",
                "doc6": "US tops 5 million confirmed virus cases"
            }
            
            # Create a mapping of document IDs to text for later retrieval
            doc_id_to_text = test_documents
            
            # Add all test documents at once
            print("\nAdding test documents in a single batch...")
            
            # Format documents for the add_documents tool
            documents_batch = []
            for doc_id, text in test_documents.items():
                documents_batch.append({
                    "id": doc_id,
                    "text": text
                })
            
            # Send all documents at once
            add_result = await self.session.call_tool(
                "add_documents",
                {
                    "documents": documents_batch
                }
            )
            
            # Print result
            for content in add_result.content:
                if content.type == "text":
                    print(f"Add documents result: {content.text}")
            
            # List all documents in the index
            print("\nListing all documents in the index:")
            list_result = await self.session.call_tool(
                "list_documents",
                {
                    "limit": 100
                }
            )
            
            # Print result
            for content in list_result.content:
                if content.type == "text":
                    result = json.loads(content.text)
                    print(f"Found {result.get('count')} documents:")
                    for doc in result.get('documents', []):
                        print(f"  ID: {doc.get('id'):<6} - {doc.get('text')[:50]}...")
            
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
                # Get the search result
                result = await self.session.call_tool(
                    "semantic_search",
                    {"query": query, "limit": 1, "graph": False}
                )
                
                try:
                    # Extract the text from the result
                    result_text = extract_text_from_result(str(result.content), doc_id_to_text)
                    
                    # Print query and best match
                    print(f"{query:<20} {result_text}")
                except Exception as e:
                    # Handle JSON parsing errors by properly escaping the content
                    try:
                        # Replace problematic escape sequences
                        content_text = result.content.text.replace("\\'", "'")
                        result = json.loads(content_text)
                        best_match = result[0]["text"] if result else "No results found"
                        print(f"{query:<20} {best_match}")
                    except Exception as e2:
                        print(f"Error parsing result: {e2}")
                        print(f"Content text: {result.content.text}")
            
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
