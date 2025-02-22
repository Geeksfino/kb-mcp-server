"""Configuration classes for data tools."""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Union, Literal

@dataclass
class ProcessorConfig:
    """Configuration for document processing."""
    chunk_size: int = 2048
    chunk_overlap: int = 20
    min_chunk_size: int = 200
    supported_extensions: List[str] = field(default_factory=lambda: ['.txt', '.md', '.pdf', '.doc', '.docx'])

@dataclass
class DatabaseConfig:
    """Configuration for txtai database."""
    # Content storage configuration
    content_url: Optional[str] = None  # SQLAlchemy URL for client database, None for SQLite
    content_schema: Optional[str] = None  # Optional schema for client databases
    
    # Vector storage configuration
    vector_backend: Literal["faiss", "pgvector", "sqlite-vec"] = "faiss"
    vector_url: Optional[str] = None  # Required for pgvector
    vector_schema: Optional[str] = None  # Optional schema for pgvector
    vector_table: str = "vectors"  # Table name for pgvector/sqlite-vec
    
    # Index configuration
    index_path: Optional[str] = None  # Path to save/load the embeddings index
    writable: bool = True  # Whether the index is writable
    
    def to_dict(self) -> Dict:
        """Convert config to txtai database configuration dictionary."""
        config = {}
        
        # Configure content storage
        if self.content_url:
            config["content"] = self.content_url
            if self.content_schema:
                config["schema"] = self.content_schema
        else:
            config["content"] = True  # Use SQLite
        
        # Configure vector storage
        if self.vector_backend == "pgvector":
            if not self.vector_url:
                raise ValueError("vector_url is required when using pgvector backend")
            config["backend"] = "pgvector"
            config["pgvector"] = {
                "url": self.vector_url,
                "table": self.vector_table
            }
            if self.vector_schema:
                config["pgvector"]["schema"] = self.vector_schema
        elif self.vector_backend == "sqlite-vec":
            config["backend"] = "sqlite"
            config["sqlite"] = {
                "table": self.vector_table
            }
        else:  # faiss
            config["backend"] = "faiss"
        
        # Add index configuration if provided
        if self.index_path:
            config["path"] = self.index_path
            config["writable"] = self.writable
            
        return config

@dataclass
class EmbeddingsConfig:
    """Configuration for txtai embeddings."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    similarity: Literal["cosine", "l2"] = "cosine"
    gpu: bool = False
    threads: int = 4
    
    def to_dict(self) -> Dict:
        """Convert config to txtai embeddings configuration dictionary."""
        return {
            "path": self.model_name,
            "similarity": self.similarity,
            "batchsize": self.batch_size,
            "gpu": self.gpu,
            "threads": self.threads
        }

@dataclass
class LoaderConfig:
    """Configuration for document loader."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    batch_size: int = 32
    progress_interval: int = 100  # Report progress every N documents
