#!/usr/bin/env python
"""
Test script for txtai semantic search functionality.
Demonstrates two approaches:
1. Direct Embeddings approach
2. Application approach using simple.yml configuration
"""

import os
import logging
import yaml
from tabulate import tabulate

from txtai.embeddings import Embeddings
from txtai.app import Application

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Test documents - same as in the reference notebook
DOCUMENTS = [
    "US tops 5 million confirmed virus cases",
    "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
    "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
    "The National Park Service warns against sacrificing slower friends in a bear attack",
    "Maine man wins $1M from $25 lottery ticket",
    "Make huge profits without work, earn up to $100,000 a day"
]

# Test queries - same as in the reference notebook
QUERIES = [
    "feel good story",
    "climate change",
    "public health story",
    "war",
    "wildlife",
    "asia",
    "lucky",
    "dishonest junk"
]

def print_results_table(results):
    """
    Print results in a clean tabular format.
    
    Args:
        results: List of [query, best_match] pairs
    """
    print(tabulate(results, headers=["Query", "Best Match"], tablefmt="grid"))


def test_with_embeddings():
    """
    Test semantic search using txtai.embeddings.Embeddings.
    This approach directly uses the Embeddings class.
    """
    logger.info("Creating embeddings with hybrid search...")
    
    # Create embeddings model with hybrid search
    embeddings = Embeddings(
        {"path": "sentence-transformers/nli-mpnet-base-v2", "hybrid": True}
    )
    
    # Index documents
    logger.info("Indexing documents...")
    embeddings.index([(i, text, None) for i, text in enumerate(DOCUMENTS)])
    
    # Run queries and collect results
    logger.info("\nQuery Results:\n----------------")
    results = []
    for query in QUERIES:
        # Extract uid of first result
        # search result format: (uid, score)
        results_query = embeddings.search(query, 1)
        logger.info(f"Query: {query}, Results: {results_query}")
        
        if results_query:
            uid = results_query[0][0]
            best_match = DOCUMENTS[uid]
            results.append([query, best_match])
        else:
            results.append([query, "No results found"])
    
    # Print results in a clean tabular format
    print_results_table(results)


def test_with_application():
    """
    Test semantic search using txtai.app.Application.
    This approach uses the Application class with a YAML configuration.
    """
    # Create application with simple.yml configuration
    config_path = os.path.join(os.path.dirname(__file__), "simple/simple.yml")
    
    # Load the configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Using configuration: {config}")
    
    # Create the Application
    app = Application(config)
    
    # Create a list of documents in the format expected by Application
    docs = []
    for i, text in enumerate(DOCUMENTS):
        # The correct format is {"id": id, "text": text} - text is the field name
        docs.append({"id": i, "text": text})
    
    # Add documents to the index
    logger.info("Adding documents to the application index...")
    app.add(docs)
    logger.info(f"Added {len(docs)} documents")
    
    # Build the index
    logger.info("Building the application index...")
    app.index()
    
    # Run queries and collect results
    logger.info("\nQuery Results:\n----------------")
    results = []
    for query in QUERIES:
        # Search returns [{"id": id, "score": score}]
        results_query = app.search(query, 1)
        logger.info(f"Query: {query}, Results: {results_query}")
        
        if results_query:
            # Get the document using the internal ID
            doc_id = results_query[0]["id"]
            if isinstance(doc_id, int) and 0 <= doc_id < len(DOCUMENTS):
                best_match = DOCUMENTS[doc_id]
            else:
                best_match = f"Unknown document ID: {doc_id}"
            
            results.append([query, best_match])
        else:
            results.append([query, "No results found"])
    
    # Format and print results
    print_results_table(results)


if __name__ == "__main__":
    # Run both tests
    test_with_embeddings()
    test_with_application()
