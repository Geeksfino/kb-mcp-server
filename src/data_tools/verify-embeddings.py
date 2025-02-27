#!/usr/bin/env python
"""
Verify an embeddings index using the TxtAISettings.from_embeddings method.

This utility script helps verify that an embeddings index is valid and working correctly.
It loads the embeddings, displays configuration information, and performs basic tests.

Usage:
    python verify_embeddings.py path/to/embeddings [--query "test query"] [--limit N]
"""

import os
import sys
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

# Add the src directory to the path if running from the repo
repo_src = Path(__file__).resolve().parents[1] / "src"
if repo_src.exists():
    sys.path.insert(0, str(repo_src))

try:
    from txtai_mcp_server.core.config import TxtAISettings
except ImportError:
    print("Error: Could not import TxtAISettings. Make sure txtai-mcp-server is installed.")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Verify an embeddings index")
    parser.add_argument("embeddings_path", help="Path to embeddings directory or archive file")
    parser.add_argument("--query", default="test query", help="Query to use for testing search")
    parser.add_argument("--limit", type=int, default=3, help="Maximum number of search results to display")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def main():
    """Verify an embeddings index."""
    args = parse_args()
    
    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(f"Verifying embeddings at: {args.embeddings_path}")
    
    try:
        # Load embeddings using TxtAISettings.from_embeddings
        print("Loading embeddings...")
        settings, app = TxtAISettings.from_embeddings(args.embeddings_path)
        print(f"Successfully loaded embeddings from: {args.embeddings_path}")
        
        # Print configuration details
        print("\nConfiguration details:")
        print(f"- Model path: {app.config.get('path', 'Not specified')}")
        if 'embeddings' in app.config:
            print(f"- Embeddings path: {app.config['embeddings'].get('path', 'Not specified')}")
            print(f"- Storage path: {app.config['embeddings'].get('storagepath', 'Not specified')}")
            print(f"- GPU enabled: {app.config['embeddings'].get('gpu', False)}")
            print(f"- Normalization: {app.config['embeddings'].get('normalize', False)}")
        
        # Check if index is available
        if hasattr(app, "embeddings") and hasattr(app.embeddings, "index"):
            print("\nEmbeddings index is available")
            if hasattr(app.embeddings.index, "count"):
                count = app.embeddings.index.count()
                print(f"Index contains {count} vectors")
        else:
            print("\nWarning: Embeddings index not found or not accessible")
        
        # Test search functionality
        print(f"\nTesting search with query: '{args.query}'")
        try:
            results = app.search(args.query, limit=args.limit)
            print(f"Search successful, found {len(results)} results")
            
            # Display results
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    text = result.get("text", str(result))
                    score = result.get("score", "N/A")
                    print(f"Result {i+1}: Score={score}, Text={text[:100]}...")
                else:
                    print(f"Result {i+1}: {str(result)[:100]}...")
        except Exception as e:
            print(f"Search failed: {str(e)}")
        
        # Test pipelines if available
        if hasattr(app, "pipelines"):
            print("\nAvailable pipelines:")
            for name in app.pipelines:
                print(f"- {name}")
            
            # Test extractor if available
            if "extractor" in app.pipelines and hasattr(app, "extract"):
                print("\nTesting extractor pipeline:")
                try:
                    # First get some search results to extract from
                    search_results = app.search(args.query, limit=1)
                    if search_results:
                        # Get text from first result
                        if isinstance(search_results[0], dict) and "text" in search_results[0]:
                            text = search_results[0]["text"]
                        else:
                            text = str(search_results[0])
                        
                        # Extract answer
                        answers = app.extract([{"text": text, "query": args.query}])
                        if answers:
                            print(f"Extraction successful: {answers[0]['answer']}")
                        else:
                            print("Extraction returned no results")
                except Exception as e:
                    print(f"Extraction failed: {str(e)}")
        
        print("\nVerification complete!")
        return 0
        
    except Exception as e:
        print(f"Error verifying embeddings: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())