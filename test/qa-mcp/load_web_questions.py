#!/usr/bin/env python
"""
Script to load the web_questions dataset into txtai embeddings.

This script:
1. Loads the web_questions dataset from Hugging Face
2. Creates a txtai embeddings index
3. Indexes the questions and answers
4. Tests the index with sample queries

Usage:
    python load_web_questions.py [--limit N] [--test] [--save-archive PATH]
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path

# Set environment variable to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
    from txtai.embeddings import Embeddings
    from txtai.app import Application
except ImportError:
    logger.error("Required packages not found. Please install with: pip install datasets txtai")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Load web_questions dataset into txtai embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of questions to load")
    parser.add_argument("--test", action="store_true", help="Run test queries after loading")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (default: qa.yml in script directory)")
    parser.add_argument("--save-archive", type=str, default=None, 
                        help="Save the embeddings as a tar.gz archive at the specified path")
    return parser.parse_args()

def load_config(config_path=None):
    """Load the txtai configuration."""
    if not config_path:
        config_path = os.path.join(os.path.dirname(__file__), "qa.yml")
    
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Get the index path from the config and make it absolute
    index_path = config.get("path", ".txtai/indexes/qa")
    if not os.path.isabs(index_path):
        index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), index_path)
    
    # Update config with absolute path
    config["path"] = index_path
    
    # Update storage path if it's not absolute
    if "embeddings" in config and "storagepath" in config["embeddings"]:
        storage_path = config["embeddings"]["storagepath"]
        if not os.path.isabs(storage_path):
            storage_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), storage_path)
            config["embeddings"]["storagepath"] = storage_path
    
    logger.info(f"Index will be stored at: {index_path}")
    
    # Create the directory if it doesn't exist
    os.makedirs(index_path, exist_ok=True)
    
    return config

def load_dataset_and_create_index(config, limit=None):
    """Load the web_questions dataset and create a txtai index."""
    logger.info("Loading web_questions dataset from Hugging Face")
    ds = load_dataset("web_questions", split="train")
    
    # Limit the dataset if requested
    if limit:
        logger.info(f"Limiting dataset to {limit} questions")
        ds = ds.select(range(min(limit, len(ds))))
    
    logger.info(f"Loaded {len(ds)} questions")
    
    # Print a few examples
    logger.info("Sample questions and answers:")
    for row in ds.select(range(min(5, len(ds)))):
        logger.info(f"Q: {row['question']}, A: {row['answers']}")
    
    # Create Application instance with the configuration
    logger.info("Creating txtai Application")
    app = Application(config)
    
    # Prepare data for indexing
    # Map question to text and store content
    logger.info("Preparing data for indexing")
    data = []
    for uid, row in enumerate(ds):
        # Join multiple answers with commas
        answer = ", ".join(row["answers"])
        
        # Create a document with ID, text, and answer
        data.append((
            f"q{uid}",  # ID
            {
                "url": row.get("url", ""),
                "text": row["question"],
                "answer": answer
            },
            None  # No tags
        ))
    
    # Index the data
    logger.info(f"Indexing {len(data)} questions")
    app.add(data)
    app.index()
    
    logger.info(f"Indexing complete. Index saved to: {config['path']}")
    
    return app

def test_index(app):
    """Test the index with sample queries."""
    logger.info("Testing the index with sample queries")
    
    test_queries = [
        "What is the timezone of New York?",
        "Who is the president of the United States?",
        "What is the capital of France?",
        "How tall is the Eiffel Tower?",
        "What is the population of California?",
        "When was the Declaration of Independence signed?",
        "What language do they speak in Brazil?",
        "What is the currency of Japan?"
    ]
    
    logger.info("%-50s %-50s %s" % ("Query", "Best Match Question", "Answer"))
    logger.info("-" * 120)
    
    for query in test_queries:
        # Use SQL to get the question and answer
        results = app.search(f"select text, answer, score from txtai where similar('{query}') limit 1")
        
        if results and len(results) > 0:
            result = results[0]
            question = result.get("text", "")
            answer = result.get("answer", "")
            score = result.get("score", 0.0)
            
            logger.info("%-50s %-50s %s (%.4f)" % (query, question, answer, score))
        else:
            logger.info("%-50s %-50s %s" % (query, "No match found", ""))
    
    # Test the extractor if available
    if hasattr(app, "pipelines") and "extractor" in app.pipelines:
        logger.info("\nTesting extractor pipeline")
        
        for query in test_queries[:3]:  # Test first 3 queries
            # First search for relevant documents
            search_results = app.search(query, limit=1)
            
            if search_results and len(search_results) > 0:
                # Get text from search result
                if isinstance(search_results[0], dict) and "text" in search_results[0]:
                    text = search_results[0]["text"]
                else:
                    text = str(search_results[0])
                
                # Extract answer
                answers = app.extract([(None, query, query, False)], [text])
                
                if answers and answers[0]:
                    logger.info(f"Query: {query}")
                    logger.info(f"Context: {text}")
                    logger.info(f"Extracted answer: {answers[0][1]}")
                    logger.info("-" * 80)

def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Load dataset and create index
        app = load_dataset_and_create_index(config, args.limit)
        
        # Save as tar.gz archive if requested
        if args.save_archive:
            archive_path = args.save_archive
            if not os.path.isabs(archive_path):
                archive_path = os.path.join(os.path.dirname(__file__), archive_path)
            
            logger.info(f"Saving embeddings as archive to: {archive_path}")
            app.embeddings.save(archive_path)
            logger.info(f"Archive saved successfully to: {archive_path}")
        
        # Test the index if requested
        if args.test:
            test_index(app)
        
        logger.info("Script completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
