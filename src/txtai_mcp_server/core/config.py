"""Configuration for txtai MCP server."""
import os
from pathlib import Path
from typing import Dict, Any, Optional, Literal

from pydantic import validator
from pydantic_settings import BaseSettings


class TxtAISettings(BaseSettings):
    """Settings for txtai MCP server.
    
    Storage Configuration:
    1. Local Storage (default):
       - Content: SQLite database in {index_path}/content.db
       - Vectors: FAISS index in {index_path}/vectors.faiss
    
    2. Remote Storage:
       - Content: PostgreSQL or other SQLAlchemy-compatible database
       - Vectors: pgvector or sqlite-vec
    """
    
    # Model settings (existing)
    model_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_gpu: bool = True
    model_normalize: bool = True
    
    # Storage settings
    storage_mode: Literal["memory", "persistence"] = "memory"
    
    # HuggingFace dataset settings (optional, for memory mode)
    dataset_enabled: bool = False  # Must be true to load dataset
    dataset_name: Optional[str] = None     # e.g. "wikipedia"
    dataset_split: str = "train"           # e.g. "train[:100]"
    dataset_text_field: Optional[str] = None  # Field containing text
    
    # Persistence settings (optional, for persistence mode)
    store_content: bool = True
    index_path: str = "~/.txtai/embeddings"  # Default path for persistence
    
    # New storage settings
    content_url: Optional[str] = None  # SQLAlchemy URL for content DB
    content_schema: Optional[str] = None  # Schema for content DB
    vector_backend: Literal["faiss", "pgvector", "sqlite-vec"] = "faiss"
    vector_url: Optional[str] = None  # URL for vector DB (required for pgvector)
    vector_schema: Optional[str] = None  # Schema for vector DB
    vector_table: str = "vectors"  # Table name for vector storage
    
    class Config:
        env_prefix = "TXTAI_"  # e.g. TXTAI_MODEL_PATH
        env_file = ".env"  # Load from .env file
        env_file_encoding = "utf-8"
    
    @validator("dataset_name")
    def validate_dataset_config(cls, v, values):
        """Validate dataset configuration."""
        if values.get("dataset_enabled", False):
            if not v:
                raise ValueError("dataset_name is required when dataset_enabled=True")
        return v
    
    @validator("index_path")
    def validate_index_path(cls, v):
        """Ensure index_path is expanded."""
        return str(Path(v).expanduser())
    
    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration dictionary."""
        config = {
            "path": self.model_path,  # Keep the HuggingFace model path
            "method": "transformers",
            "transform": "mean",
            "normalize": self.model_normalize,
            "gpu": self.model_gpu
        }
        
        # Configure content storage
        if self.content_url:
            config["content"] = self.content_url
            if self.content_schema:
                config["schema"] = self.content_schema
        else:
            config["content"] = self.store_content
        
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
            config["backend"] = "sqlite-vec"
            config["table"] = self.vector_table
        
        # Set storage path for local storage
        if self.index_path:
            config["storage"] = self.index_path
        elif not self.content_url and self.vector_backend == "faiss":
            # Default local storage path
            config["storage"] = str(Path.home() / ".txtai" / "embeddings")
        
        return config
    
    @classmethod
    def load(cls) -> "TxtAISettings":
        """Load settings from environment and .env file."""
        # Find the project root (where .env should be)
        current_dir = Path(__file__).parent
        while current_dir.name != "embedding-mcp-server" and current_dir.parent != current_dir:
            current_dir = current_dir.parent
        
        env_path = current_dir / ".env"
        return cls(_env_file=str(env_path) if env_path.exists() else None)
