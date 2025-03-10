#!/usr/bin/env python3
"""Configuration utilities for data tools."""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import yaml
from txtai.app import Application

logger = logging.getLogger(__name__)

def find_config_file() -> Optional[str]:
    """
    Find a configuration file in standard locations.
    
    Returns:
        Path to the configuration file if found, None otherwise.
    """
    # Check environment variable first
    if os.environ.get("KNOWLEDGE_BASE_CONFIG"):
        config_path = os.environ.get("KNOWLEDGE_BASE_CONFIG")
        if os.path.exists(config_path):
            return config_path
    
    # Check standard locations
    search_paths = [
        "./config.yaml",
        "./config.yml",
        Path.home() / ".config" / "knowledge-base" / "config.yaml",
        Path.home() / ".config" / "knowledge-base" / "config.yml",
        Path.home() / ".knowledge-base" / "config.yaml",
        Path.home() / ".knowledge-base" / "config.yml",
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return str(path)
    
    return None

def create_default_config() -> Dict[str, Any]:
    """
    Create a default configuration.
    
    Returns:
        Default configuration dictionary.
    """
    return {
        "path": ".txtai/knowledge-base",
        "content": True,
        "writable": True,
        "embeddings": {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "normalize": True,
            "gpu": True,
            "storagetype": "annoy",
            "storagepath": ".txtai/embeddings",
            "content": True
        },
        "processor": {
            "segmentation": "paragraphs",  # Options: sentences, paragraphs, sections
            "minlength": 100,
            "maxlength": 1000,
            "cleantext": True,
            "late_chunking": False,  # Enable/disable late chunking strategy
            "chunk_size": 1000,      # Maximum size of chunks when using late chunking
            "chunk_overlap": 100,    # Overlap between chunks when using late chunking
            "bypass_textractor": False,  # Bypass Textractor for simple file types
            "segmenter": {
                "type": "splitter",
                "sentences": True,
                "minlength": 100,
                "maxlength": 1000
            }
        },
        "graph": {
            "similarity": 0.75,
            "limit": 10
        },
        "pipeline": {
            "ner": {
                "path": "NER",
                "method": "entity"
            }
        }
    }

def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save a configuration to a file.
    
    Args:
        config: Configuration dictionary
        path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved configuration to {path}")
