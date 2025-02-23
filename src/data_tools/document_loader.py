#!/usr/bin/env python3
"""Command-line interface for document loading."""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Generator, Dict
import json

from datasets import load_dataset
from .config import LoaderConfig, DatabaseConfig, EmbeddingsConfig
from .loader_utils import DocumentLoader

logger = logging.getLogger(__name__)

def process_huggingface_dataset(dataset_name: str, split: str, text_field: Optional[str] = None) -> Generator[Dict[str, str], None, None]:
    """Process a HuggingFace dataset into document chunks.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        split: Dataset split (e.g., "train[:100]")
        text_field: Field to use as text content, auto-detected if None
        
    Yields:
        Dict containing text chunk and metadata
        
    Raises:
        ValueError: If text field is not found in dataset
        Exception: If dataset loading fails
    """
    try:
        logger.info(f"Loading dataset {dataset_name} (split: {split})")
        dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Dataset loaded successfully with {len(dataset)} items")
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise
    
    # Validate dataset is not empty
    if len(dataset) == 0:
        raise ValueError(f"Dataset {dataset_name} (split: {split}) is empty")
    
    # Get sample item to check fields
    sample_item = dataset[0]
    available_fields = list(sample_item.keys())
    
    # Auto-detect text field if not specified
    if text_field is None:
        text_fields = {
            "web_questions": "question",
            "squad": "question",
            "eli5": "title",
            "wikipedia": "text",
            "bookcorpus": "text"
        }
        text_field = text_fields.get(dataset_name.split('/')[-1], "text")
        logger.info(f"Auto-detected text field: {text_field}")
    
    # Validate text field exists
    if text_field not in available_fields:
        raise ValueError(
            f"Text field '{text_field}' not found in dataset. "
            f"Available fields: {available_fields}"
        )
    
    # Check if field contains text data
    sample_text = sample_item[text_field]
    if not isinstance(sample_text, str):
        raise ValueError(
            f"Text field '{text_field}' contains {type(sample_text).__name__}, "
            f"expected string. Value: {sample_text}"
        )
    
    processed = 0
    skipped = 0
    for i, item in enumerate(dataset):
        text = item.get(text_field)
        if not text or not isinstance(text, str):
            skipped += 1
            continue
            
        processed += 1
        # Convert metadata to JSON string for txtai
        metadata = json.dumps({
            "source": dataset_name,
            "split": split,
            "index": i
        })
        yield {
            "id": f"{dataset_name}-{i}",
            "text": text,
            "metadata": metadata
        }
    
    # Log processing summary
    logger.info(
        f"Dataset processing complete: {processed} items processed, "
        f"{skipped} items skipped"
    )
    if skipped > 0:
        logger.warning(
            f"Skipped {skipped} items due to missing or invalid text in "
            f"field '{text_field}'"
        )

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Load documents into txtai database")
    parser.add_argument("--dataset", help="HuggingFace dataset to load")
    parser.add_argument("--split", help="Dataset split to load")
    parser.add_argument("--text-field", help="Field to use as text content")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level)
    
    try:
        # Create loader using MCP server settings
        loader = DocumentLoader()
        
        # Load dataset
        if not args.dataset or not args.split:
            raise ValueError("Both --dataset and --split are required")
            
        # Process dataset
        loader.process_huggingface_dataset(
            dataset_name=args.dataset,
            split=args.split,
            text_field=args.text_field
        )
        
        # Try a test search
        results = loader.search("test query", limit=3)
        logger.info("Search results:")
        for result in results:
            logger.info(f"- Text: {result['text']}")
            logger.info(f"  Score: {result['score']:.4f}")
            
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
