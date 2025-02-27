#!/usr/bin/env python
"""
Direct test of txtai Embeddings API to understand document indexing behavior.
This test uses the Embeddings class directly instead of the Application class.
"""

import os
import yaml
import shutil

# Set environment variable to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from txtai import Embeddings

def main():
    # Load the same configuration
    config_path = os.path.join(os.path.dirname(__file__), "./", "simple.yml")
    print(f"Loading configuration from: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Extract embeddings configuration
    embeddings_config = config.get("embeddings", {})
    print(f"Embeddings configuration: {embeddings_config}")
    
    # Get the index path from the config
    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.get("path", ".txtai/indexes/simple"))
    print(f"Index will be stored at: {index_path}")
    
    # Create the directory if it doesn't exist
    os.makedirs(index_path, exist_ok=True)
    
    # Create Embeddings instance with path to save the index
    embeddings = Embeddings(
        path=embeddings_config.get("path", "sentence-transformers/nli-mpnet-base-v2"),
        # Use the content setting from the config file
        storagetype="sqlite",  # Use sqlite for persistent storage
        storagepath=index_path,
        hybrid=True,  # Enable hybrid search to match the Application API
        content=True  # Explicitly enable content storage
    )
    
    # Test documents
    test_documents = [
        {"id": "doc1", "text": "Maine man wins $1M from $25 lottery ticket"},
        {"id": "doc2", "text": "Make huge profits without work, earn up to $100,000 a day"},
        {"id": "doc3", "text": "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg"},
        {"id": "doc4", "text": "Beijing mobilises invasion craft along coast as Taiwan tensions escalate"},
        {"id": "doc5", "text": "The National Park Service warns against sacrificing slower friends in a bear attack"},
        {"id": "doc6", "text": "US tops 5 million confirmed virus cases"}
    ]
    
    # Create a mapping from document ID to index
    id_to_index = {doc["id"]: i for i, doc in enumerate(test_documents)}
    
    # Create data in the format expected by Embeddings
    # Embeddings expects (id, text, tags) format or just text
    data = [(doc["id"], doc["text"], None) for doc in test_documents]
    
    # Index the data
    print("\nIndexing data with Embeddings API directly:")
    embeddings.index(data)
    
    # Save the index to disk
    print(f"Saving index to: {index_path}")
    embeddings.save(index_path)
    
    # Test search
    print("\nTesting search with Embeddings API:")
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
    
    print("%-20s %s" % ("Query", "Best Match"))
    print("-" * 50)
    
    for query in test_queries:
        # Search returns (id, score) tuples
        results = embeddings.search(query, 1)
        if results:
            # Handle different result formats
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], tuple) and len(results[0]) == 2:
                    # Format: [(id, score)]
                    result_id = results[0][0]
                elif isinstance(results[0], dict) and 'id' in results[0]:
                    # Format: [{'id': id, 'score': score}]
                    result_id = results[0]['id']
                else:
                    print(f"%-20s %s" % (query, f"Unknown result format: {results}"))
                    continue
                
                # Find the document with this ID
                for doc in test_documents:
                    if doc["id"] == result_id:
                        print("%-20s %s" % (query, doc["text"]))
                        break
                else:
                    print("%-20s %s" % (query, f"Unknown ID: {result_id}"))
            else:
                print(f"%-20s %s" % (query, f"Unexpected result format: {results}"))
        else:
            print("%-20s %s" % (query, "No results"))
    
    # Test direct ID lookup
    print("\nTesting direct ID lookup with Embeddings API:")
    for doc_id in ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]:
        # Try to find the document by ID
        found = False
        for i, (id_, text, _) in enumerate(data):
            if id_ == doc_id:
                print(f"ID lookup for {doc_id}: Found at index {i} with text: {text[:50]}...")
                found = True
                break
        
        if not found:
            print(f"ID lookup for {doc_id}: Not found")
    
    # Test similarity search by ID
    print("\nTesting similarity search by ID with Embeddings API:")
    for doc_id in ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"]:
        # Find the document text for this ID
        doc_text = None
        for doc in test_documents:
            if doc["id"] == doc_id:
                doc_text = doc["text"]
                break
        
        if doc_text:
            # Get similar documents to this one using the text
            similar = embeddings.search(doc_text, 3)
            print(f"Documents similar to {doc_id}:")
            
            for result in similar:
                # Handle different result formats
                if isinstance(result, dict) and 'id' in result:
                    similar_id = result['id']
                    score = result.get('score', 0.0)
                elif isinstance(result, tuple) and len(result) == 2:
                    similar_id, score = result
                else:
                    print(f"  - Unknown result format: {result}")
                    continue
                
                # Find the document with this ID
                for doc in test_documents:
                    if doc["id"] == similar_id:
                        print(f"  - {similar_id} (Score: {score:.4f}): {doc['text'][:50]}...")
                        break
                else:
                    print(f"  - Unknown ID: {similar_id} (Score: {score:.4f})")
        else:
            print(f"Could not find document with ID {doc_id}")

if __name__ == "__main__":
    main()
