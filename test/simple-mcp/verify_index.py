#!/usr/bin/env python
"""
Script to verify the state of the txtai index.
"""

import os
import sys
import yaml
import json
import glob
import struct

# Set environment variable to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from txtai.app import Application

def main():
    """Verify the txtai index state."""
    # Load the same configuration as the server
    config_path = os.path.join(os.path.dirname(__file__), "./", "simple.yml")
    print(f"Loading configuration from: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print("Configuration:")
    print(yaml.dump(config))
    
    # Check index directory files directly
    index_path = os.path.expanduser(os.path.join(config.get("path", ".txtai/indexes/simple")))
    print(f"\nExamining index directory: {index_path}")
    
    if os.path.exists(index_path):
        # List index files
        files = glob.glob(os.path.join(index_path, "*"))
        print(f"Found {len(files)} files in index directory:")
        
        for file in files:
            file_name = os.path.basename(file)
            file_size = os.path.getsize(file)
            print(f"  {file_name:<15} - {file_size} bytes")
        
        # Try to read IDs file
        ids_file = os.path.join(index_path, "ids")
        if os.path.exists(ids_file):
            try:
                with open(ids_file, "rb") as f:
                    content = f.read()
                    print(f"\nIDs file content (hex): {content.hex()}")
                    print(f"IDs file content (raw): {content}")
                    
                    # Try to interpret as string
                    try:
                        as_str = content.decode('utf-8', errors='replace')
                        print(f"IDs file as string: {repr(as_str)}")
                    except Exception as e:
                        print(f"Error decoding as string: {e}")
            except Exception as e:
                print(f"Error reading IDs file: {e}")
    else:
        print(f"Index directory {index_path} does not exist")
    
    # Create Application instance with the same config
    app = Application(config)
    
    # Check if the index exists and has documents
    try:
        # Get the raw index information if possible
        if hasattr(app, "embeddings") and hasattr(app.embeddings, "backend"):
            print("\nEmbeddings backend information:")
            backend = app.embeddings.backend
            print(f"  Backend type: {type(backend).__name__}")
            
            # Try to access count information
            if hasattr(backend, "count"):
                print(f"  Backend document count: {backend.count()}")
        
        # Search with wildcard to get all documents
        results = app.search("*", 100)
        print(f"\nFound {len(results)} documents in the index using wildcard search")
        
        # Print document IDs
        if results:
            print("\nDocument IDs from wildcard search:")
            for i, result in enumerate(results[:10]):  # Show first 10
                if isinstance(result, dict):
                    doc_id = result.get("id", "unknown")
                    score = result.get("score", 0)
                    print(f"  {i+1}. ID: {doc_id}, Score: {score}")
                else:
                    print(f"  {i+1}. {result}")
            
            if len(results) > 10:
                print(f"  ... and {len(results) - 10} more")
        
        # Test search with specific queries
        test_queries = [
            "feel good story",
            "climate change",
            "public health story",
            "war",
            "wildlife",
            "asia",
            "lucky",
            "dishonest junk"
        ]
        
        print("\nTesting search with specific queries:")
        for query in test_queries:
            results = app.search(query, 1)
            print(f"  Query: {query:<20} -> ", end="")
            if results:
                if isinstance(results[0], dict):
                    doc_id = results[0].get("id", "unknown")
                    text = results[0].get("text", "")[:50] + "..." if "text" in results[0] else ""
                    print(f"Document ID: {doc_id} - {text}")
                else:
                    print(f"Result: {results[0]}")
            else:
                print("No results")
                
        def escape_sql_string(text: str) -> str:
            """Escape a string for use in SQL queries."""
            if text is None:
                return text
            return text.replace("'", "''")

        print("\nTrying direct ID searches:")
        for doc_id in ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8"]:
            try:
                # Try direct embeddings backend access first
                if hasattr(app, "embeddings") and hasattr(app.embeddings, "backend"):
                    try:
                        # Get document directly from backend
                        result = app.embeddings.backend.get(doc_id)
                        print(f"  ID: {doc_id:<6} -> Backend: ", end="")
                        if result is not None:
                            print(f"Found document in backend")
                        else:
                            print("Not found in backend")
                    except Exception as e:
                        print(f"Error accessing backend for ID {doc_id}: {e}")
        
                # Try getting document from cache
                try:
                    from txtai_mcp_server.core.state import get_document_from_cache
                    text = get_document_from_cache(doc_id)
                    print(f"  ID: {doc_id:<6} -> Cache: ", end="")
                    if text:
                        print(f"Found document with text: {text[:50]}...")
                    else:
                        print("Not found in cache")
                except Exception as e:
                    print(f"Error accessing cache for ID {doc_id}: {e}")
        
                # Try SQL syntax for direct ID lookup with proper escaping
                try:
                    # First try to get all documents
                    escaped_id = escape_sql_string(doc_id)
                    results = app.search(f"select id, text from txtai where id = '{escaped_id}' and text is not null")
                    print(f"  ID: {doc_id:<6} -> SQL query: ", end="")
                    
                    if results and len(results) > 0:
                        if isinstance(results[0], dict):
                            found_id = results[0].get("id")
                            text = results[0].get("text", "")[:50]
                            if found_id == doc_id:  # Make sure we got the right document
                                print(f"Found document with ID: {found_id}, text: {text}...")
                            else:
                                print(f"Warning: Got wrong document ID: {found_id}")
                        else:
                            print(f"Result in unexpected format: {results[0]}")
                    else:
                        print("No results")
                except Exception as e:
                    print(f"Error with SQL query for ID {doc_id}: {e}")
            except Exception as e:
                print(f"Error searching for ID {doc_id}: {e}")
                
        # Try a different search approach if possible
        try:
            print("\nTrying alternative direct query approach:")
            
            # Test direct txtai API
            print("\nTesting direct txtai API:")
            # Create a fresh instance with test documents
            test_app = Application({
                "embeddings": {
                    "path": "sentence-transformers/all-MiniLM-L6-v2",
                    "content": True
                },
                "writable": True  # Enable index writing
            })
            
            # No need to call configure again since we passed the config to the constructor
            
            # Add test documents
            test_documents = [
                {"id": "test1", "text": "This is test document 1"},
                {"id": "test2", "text": "This is test document 2"},
                {"id": "test3", "text": "This is test document 3"}
            ]
            
            print(f"Adding {len(test_documents)} test documents")
            test_app.add(test_documents)
            test_app.index()
            
            # Test direct ID lookup
            print("\nTesting direct ID lookup with fresh instance:")
            for doc_id in ["test1", "test2", "test3"]:
                # Try SQL query
                sql_results = test_app.search(f"select * from txtai where id = '{doc_id}'")
                print(f"  ID: {doc_id} -> ", end="")
                if sql_results and len(sql_results) > 0:
                    found_id = sql_results[0].get("id") if isinstance(sql_results[0], dict) else "unknown"
                    if found_id == doc_id:
                        print(f"Success! Found correct document")
                    else:
                        print(f"Error: Found wrong document with ID {found_id}")
                else:
                    print("No results")
        except Exception as e:
            print(f"Error accessing embeddings data: {e}")
    
    except Exception as e:
        print(f"Error accessing index: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()