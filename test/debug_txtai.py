#!/usr/bin/env python
"""
Debug script to understand how txtai Application handles document IDs.
"""

import os
import sys
import yaml
from txtai.app import Application

def main():
    """Main function to test txtai Application."""
    # Load the same configuration as the server
    config_path = os.path.join(os.path.dirname(__file__), "simple", "simple.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    print("Using configuration:")
    print(yaml.dump(config))
    
    # Create Application instance
    app = Application(config)
    
    # Add documents with string IDs
    documents = [
        {"id": "doc1", "text": "Maine man wins $1M from $25 lottery ticket"},
        {"id": "doc2", "text": "Climate change creating ideal conditions for disease-spreading mosquitoes"},
        {"id": "doc3", "text": "New COVID variant cases rising in multiple states"},
        {"id": "doc4", "text": "Beijing mobilises invasion craft along coast as Taiwan tensions escalate"},
        {"id": "doc5", "text": "The National Park Service warns against sacrificing slower friends in a bear attack"},
        {"id": "doc6", "text": "Dishonest reporter fired after fabricating quotes in story"}
    ]
    
    # Print documents before adding
    print("\nDocuments to add:")
    for doc in documents:
        print(f"  {doc}")
    
    # Add documents
    app.add(documents)
    
    # Build the index
    app.index()
    
    # Create a document cache
    document_cache = {}
    for doc in documents:
        document_cache[doc["id"]] = doc["text"]
    
    # Print document cache
    print("\nDocument cache:")
    for key, value in document_cache.items():
        print(f"  {key}: {value[:30]}...")
    
    # Search for documents
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
    
    print("\nSearch results:")
    for query in queries:
        results = app.search(query, limit=1)
        print(f"\nQuery: {query}")
        print(f"Raw results: {results}")
        
        if results:
            # Try to get the document text from the cache
            doc_id = results[0]["id"]
            text = document_cache.get(doc_id, "No text available")
            print(f"Document ID: {doc_id}, Type: {type(doc_id)}")
            print(f"Text: {text[:50]}...")
        else:
            print("No results found")

if __name__ == "__main__":
    main()
