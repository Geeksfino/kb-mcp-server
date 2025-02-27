#!/usr/bin/env python
"""
Test script to verify that the QA embeddings work with the MCP server.

This script:
1. Loads the QA embeddings
2. Creates a txtai Application instance
3. Tests the answer_question functionality directly

Usage:
    python test_qa_server.py [--config PATH]
    python test_qa_server.py [--embeddings PATH]
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path

# Set environment variable to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

try:
    from txtai.app import Application
    from txtai.embeddings import Embeddings
    from txtai_mcp_server.core.state import set_txtai_app, get_txtai_app
except ImportError:
    logger.error("Required packages not found. Make sure the embedding-mcp-server is in your Python path.")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test QA embeddings with MCP server")
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to config file (default: qa.yml in script directory)")
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Path to embeddings directory or archive file (overrides config file)")
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
    
    logger.info(f"Index will be loaded from: {index_path}")
    
    return config

def test_qa_functionality(app):
    """Test the QA functionality directly."""
    logger.info("Testing QA functionality directly")
    
    test_questions = [
        "What is the timezone of New York?",
        "Who is the president of the United States?",
        "What is the capital of France?",
        "How tall is the Eiffel Tower?",
        "What is the population of California?",
        "When was the Declaration of Independence signed?",
        "What language do they speak in Brazil?",
        "What is the currency of Japan?"
    ]
    
    logger.info("%-50s %s" % ("Question", "Answer"))
    logger.info("-" * 80)
    
    for question in test_questions:
        # First approach: Try using extractor pipeline if available
        if hasattr(app, "pipelines") and "extractor" in getattr(app, "pipelines", {}) and hasattr(app, "extract"):
            logger.info(f"Using extractor pipeline for: {question}")
            
            # First search for relevant documents
            search_results = app.search(question, limit=3)
            
            if search_results:
                # Extract texts from search results
                texts = []
                for result in search_results:
                    if isinstance(result, dict) and "text" in result:
                        texts.append(result["text"])
                
                if texts:
                    # Create extraction queue
                    queue = [(None, question, question, False)]
                    
                    # Extract answers
                    answers = app.extract(queue, texts)
                    
                    if answers and answers[0] and len(answers[0]) > 1:
                        logger.info("%-50s %s" % (question, answers[0][1]))
                        continue
            
            logger.info("Extractor pipeline failed, falling back to search")
        
        # Second approach: Use SQL-based search
        try:
            # Escape single quotes in the question
            safe_question = question.replace("'", "''")
            
            # Try to get the answer field if it exists
            sql_query = f"select text, answer, score from txtai where similar('{safe_question}') limit 1"
            results = app.search(sql_query)
            
            if results and len(results) > 0 and "answer" in results[0]:
                logger.info("%-50s %s" % (question, results[0]["answer"]))
                continue
        except Exception as e:
            logger.info(f"Error getting answer field: {str(e)}, falling back to text field")
        
        # Third approach: Just return the most similar text
        results = app.search(question, limit=1)
        
        if results and len(results) > 0:
            if isinstance(results[0], dict):
                logger.info("%-50s %s" % (question, results[0].get("text", "No answer found")))
            else:
                logger.info("%-50s %s" % (question, str(results[0])))
        else:
            logger.info("%-50s %s" % (question, "No answer found"))

def main():
    """Main function."""
    args = parse_args()
    
    try:
        # Check if embeddings path is provided directly
        if args.embeddings:
            # Convert to absolute path if needed
            embeddings_path = args.embeddings
            if not os.path.isabs(embeddings_path):
                # Check if the path is relative to the current directory
                if os.path.exists(embeddings_path):
                    embeddings_path = os.path.abspath(embeddings_path)
                # Check if the path is relative to the test directory
                elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), embeddings_path)):
                    embeddings_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), embeddings_path)
                # Check if the path is relative to the project root
                elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), embeddings_path)):
                    embeddings_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), embeddings_path)
            
            logger.info(f"Loading embeddings directly from: {embeddings_path}")
            
            # Check if the path exists
            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(f"Embeddings path not found: {embeddings_path}")
            
            # Check if it's a directory or a file
            if os.path.isdir(embeddings_path):
                # For directories, create an Embeddings instance and load it
                embeddings = Embeddings()
                embeddings.load(embeddings_path)
                
                # Create an Application with the loaded embeddings
                app = Application({})
                app.embeddings = embeddings
            else:
                # For files (like tar.gz), use Application constructor with YAML string
                logger.info(f"Loading embeddings from archive file: {embeddings_path}")
                app = Application(f"path: {embeddings_path}")
        else:
            # Load configuration
            config = load_config(args.config)
            
            # Create Application instance with the configuration
            logger.info("Creating txtai Application from config")
            app = Application(config)
        
        # Set the global txtai app
        set_txtai_app(app)
        
        # Print configuration details
        logger.info("\nConfiguration details:")
        logger.info(f"- Model path: {app.config.get('path', 'Not specified')}")
        if hasattr(app, 'embeddings') and app.embeddings and hasattr(app.embeddings, 'config'):
            logger.info(f"- Embeddings path: {app.embeddings.config.get('path', 'Not specified')}")
            logger.info(f"- Storage path: {app.embeddings.config.get('storagepath', 'Not specified')}")
            logger.info(f"- GPU enabled: {app.embeddings.config.get('gpu', False)}")
        elif 'embeddings' in app.config:
            logger.info(f"- Embeddings path: {app.config['embeddings'].get('path', 'Not specified')}")
            logger.info(f"- Storage path: {app.config['embeddings'].get('storagepath', 'Not specified')}")
            logger.info(f"- GPU enabled: {app.config['embeddings'].get('gpu', False)}")
        
        # Test the QA functionality
        test_qa_functionality(app)
        
        logger.info("\nTest completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
