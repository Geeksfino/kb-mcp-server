#!/usr/bin/env python3
"""
Script to demonstrate proper document processing and indexing with txtai Application.
"""

import yaml
import os
from txtai.app import Application

def main():
    # Path to the document
    doc_path = "../knowledgebase/data_science.pdf"
    
    # Load the configuration
    config_path = "./kb.yml"
    
    print(f"Using configuration from: {config_path}")
    
    # Create application
    app = Application(config_path)
    
    # Clear the existing index by deleting the index directory
    index_path = app.config.get("path", ".txtai")
    if os.path.exists(index_path):
        print(f"Clearing existing index at: {index_path}")
    
    print(f"Processing document: {doc_path}")
    
    # Extract text segments using the textractor pipeline
    segments = list(app.pipelines["textractor"](doc_path))
    print(f"Extracted {len(segments)} segments")
    
    # Print a sample of the first segment to verify content
    if segments:
        print("\nSample of first segment:")
        print(segments[0][:200] + "...\n")
    
    # Create documents with proper structure
    documents = []
    for i, text in enumerate(segments):
        doc_id = f"data_science_{i}"
        documents.append({
            "id": doc_id,
            "text": text,
            "metadata": {
                "source": doc_path,
                "index": i
            }
        })
    
    # Add documents to the index
    print("Adding documents to index...")
    app.add(documents)
    
    # Build the index
    print("Building index...")
    app.index()
    
    # Run a test search
    query = "What tools are used for data visualization?"
    print(f"\nSearching for: '{query}'")
    
    results = app.search(query, limit=5)
    
    # Print results
    print("\nSearch results:")
    for result in results:
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:200]}...")
        print("-" * 80)

if __name__ == "__main__":
    main()
