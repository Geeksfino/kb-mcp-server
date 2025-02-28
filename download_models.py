#!/usr/bin/env python3
"""
Script to download Hugging Face models during Docker build.
This pre-caches models in the container for faster startup.
If models are already in the local cache, they won't be downloaded again.
"""

import argparse
import os
import sys
from typing import List
from pathlib import Path


def get_huggingface_cache_dir() -> Path:
    """Get the Hugging Face cache directory."""
    try:
        from huggingface_hub import constants
        return Path(constants.HF_HUB_CACHE)
    except ImportError:
        # Default cache location if huggingface_hub is not installed yet
        cache_home = os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        return Path(cache_home) / "huggingface" / "hub"


def check_model_in_cache(model_name: str, cache_dir: Path) -> bool:
    """Check if a model is already in the cache."""
    # Convert model name to directory format (replace '/' with '--')
    model_dir_name = model_name.replace('/', '--')
    
    # Check if the model directory exists in the cache
    model_dir = cache_dir / "models--" + model_dir_name
    
    # Check if the directory exists and contains model files
    if model_dir.exists() and any(model_dir.glob("**/pytorch_model.bin")):
        return True
    
    # Also check for safetensors format
    if model_dir.exists() and any(model_dir.glob("**/model.safetensors")):
        return True
    
    return False


def download_transformers_model(model_name: str) -> bool:
    """Download a Hugging Face Transformers model if not already in cache."""
    try:
        from transformers import AutoModel, AutoTokenizer
        
        # Check if model is already in cache
        cache_dir = get_huggingface_cache_dir()
        if check_model_in_cache(model_name, cache_dir):
            print(f"Model {model_name} already in cache, skipping download")
            return True
        
        print(f"Downloading model: {model_name}")
        
        # Download the model
        AutoModel.from_pretrained(model_name)
        
        # Download the tokenizer
        AutoTokenizer.from_pretrained(model_name)
        
        print(f"Successfully downloaded model: {model_name}")
        return True
    except Exception as e:
        print(f"Error downloading model {model_name}: {str(e)}")
        return False


def download_sentence_transformers_model(model_name: str) -> bool:
    """Download a Sentence Transformers model if not already in cache."""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Check if model is already in cache
        cache_dir = get_huggingface_cache_dir()
        if check_model_in_cache(model_name, cache_dir):
            print(f"Sentence Transformer model {model_name} already in cache, skipping download")
            return True
        
        print(f"Downloading Sentence Transformers model: {model_name}")
        
        # Download the model
        SentenceTransformer(model_name)
        
        print(f"Successfully downloaded Sentence Transformers model: {model_name}")
        return True
    except Exception as e:
        print(f"Error downloading Sentence Transformers model {model_name}: {str(e)}")
        return False


def parse_model_list(models_str: str) -> List[str]:
    """Parse comma-separated model list into a list of model names."""
    if not models_str:
        return []
    
    return [model.strip() for model in models_str.split(',') if model.strip()]


def main():
    parser = argparse.ArgumentParser(description='Download Hugging Face models')
    parser.add_argument('--transformers', type=str, help='Comma-separated list of Transformers models to download')
    parser.add_argument('--sentence-transformers', type=str, help='Comma-separated list of Sentence Transformers models to download')
    
    args = parser.parse_args()
    
    # Process Transformers models
    transformers_models = parse_model_list(args.transformers)
    for model in transformers_models:
        success = download_transformers_model(model)
        if not success:
            sys.exit(1)
    
    # Process Sentence Transformers models
    sentence_transformers_models = parse_model_list(args.sentence_transformers)
    for model in sentence_transformers_models:
        success = download_sentence_transformers_model(model)
        if not success:
            sys.exit(1)
    
    print("All models processed successfully!")


if __name__ == "__main__":
    main()
