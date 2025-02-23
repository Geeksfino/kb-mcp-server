"""Configuration classes for data tools."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Union, Literal
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class ProcessorConfig:
    """Configuration for document processor."""
    chunk_size: int = 1000  # Number of characters per chunk
    chunk_overlap: int = 100  # Number of characters to overlap between chunks

@dataclass
class DatabaseConfig:
    """Configuration for database storage."""
    path: str = ".txtai/embeddings"  # Path for embeddings database
    
    def __post_init__(self):
        """Post initialization hook."""
        # Convert relative path to absolute
        if self.path and not os.path.isabs(self.path):
            self.path = os.path.join(os.path.expanduser("~"), self.path)

@dataclass
class EmbeddingsConfig:
    """Configuration for txtai embeddings."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Use sentence transformers model
    method: str = "sentence-transformers"  # Use sentence transformers method
    batch_size: int = 32  # Batch size for processing

@dataclass
class LoaderConfig:
    """Configuration for document loader."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    batch_size: int = 32  # Number of documents to process at once
