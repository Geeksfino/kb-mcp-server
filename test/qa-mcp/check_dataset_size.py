#!/usr/bin/env python
"""
Script to check the size of the web_questions dataset.
"""

import os
import logging
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Check the size of the web_questions dataset."""
    logger.info("Loading web_questions dataset from Hugging Face")
    ds = load_dataset("web_questions", split="train")
    
    logger.info(f"Dataset size: {len(ds)} questions")
    
    # Print a few examples
    logger.info("Sample questions and answers:")
    for row in ds.select(range(min(5, len(ds)))):
        logger.info(f"Q: {row['question']}, A: {row['answers']}")
    
    return 0

if __name__ == "__main__":
    main()
