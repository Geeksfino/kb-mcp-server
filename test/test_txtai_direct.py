#!/usr/bin/env python
"""
Direct test of txtai API to understand document indexing behavior.
"""

import os
import yaml
from txtai.app import Application

def main():
    # Load the same configuration
    config_path = os.path.join(os.path.dirname(__file__), "simple", "simple.yml")
    print(f"Loading configuration from: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create Application instance
    app = Application(config)
    
    # Test documents
    test_documents = [
        {"id": "doc1", "text": "Maine man wins $1M from $25 lottery ticket"},
        {"id": "doc2", "text": "Make huge profits without work, earn up to $100,000 a day"},
        {"id": "doc3", "text": "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg"},
        {"id": "doc4", "text": "Beijing mobilises invasion craft along coast as Taiwan tensions escalate"},
        {"id": "doc5", "text": "The National Park Service warns against sacrificing slower friends in a bear attack"},
        {"id": "doc6", "text": "US tops 5 million confirmed virus cases"}
    ]
    
    # Try both dictionary and tuple formats
    print("\nTesting with dictionary format:")
    app.add(test_documents)
    app.index()
    
    # Check the index
    results = app.search("*", 100)
    print(f"Found {len(results)} documents with wildcard search")
    for i, result in enumerate(results[:10]):
        if isinstance(result, dict):
            print(f"  {i+1}. ID: {result.get('id', 'unknown')}, Score: {result.get('score', 0)}")
        else:
            print(f"  {i+1}. {result}")
    
    # Try direct ID lookup
    print("\nTrying direct ID lookups with dictionary format:")
    for doc_id in ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]:
        try:
            results = app.search(f"select * from txtai where id = '{doc_id}' limit 1")
            print(f"ID lookup for {doc_id}: {results}")
        except Exception as e:
            print(f"Error with ID lookup for {doc_id}: {e}")
    
    # Try with tuple format
    print("\nTesting with tuple format:")
    app = Application(config)  # Create a new instance
    
    # Convert to tuples: (id, text, metadata)
    tuple_docs = [(doc["id"], doc["text"], None) for doc in test_documents]
    app.add(tuple_docs)
    app.index()
    
    # Check the index
    results = app.search("*", 100)
    print(f"Found {len(results)} documents with wildcard search")
    for i, result in enumerate(results[:10]):
        if isinstance(result, dict):
            print(f"  {i+1}. ID: {result.get('id', 'unknown')}, Score: {result.get('score', 0)}")
        else:
            print(f"  {i+1}. {result}")
    
    # Try direct ID lookup
    print("\nTrying direct ID lookups with tuple format:")
    for doc_id in ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]:
        try:
            results = app.search(f"select * from txtai where id = '{doc_id}' limit 1")
            print(f"ID lookup for {doc_id}: {results}")
        except Exception as e:
            print(f"Error with ID lookup for {doc_id}: {e}")

if __name__ == "__main__":
    main()
