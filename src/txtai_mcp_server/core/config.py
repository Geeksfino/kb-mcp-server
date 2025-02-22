"""Configuration for txtai MCP server."""
import os
from pathlib import Path
from typing import Dict, Any

from pydantic_settings import BaseSettings


class TxtAISettings(BaseSettings):
    """Settings for txtai MCP server."""
    
    # Model settings
    model_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_gpu: bool = True
    model_normalize: bool = True
    
    # Storage settings
    store_content: bool = True
    
    class Config:
        env_prefix = "TXTAI_"  # e.g. TXTAI_MODEL_PATH
        env_file = ".env"  # Load from .env file
        env_file_encoding = "utf-8"
        
    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration dictionary."""
        return {
            "path": self.model_path,
            "method": "transformers",
            "transform": "mean",
            "normalize": self.model_normalize,
            "content": self.store_content,
            "gpu": self.model_gpu
        }
    
    @classmethod
    def load(cls) -> "TxtAISettings":
        """Load settings from environment and .env file."""
        # Find the project root (where .env should be)
        current_dir = Path(__file__).parent
        while current_dir.name != "embedding-mcp-server" and current_dir.parent != current_dir:
            current_dir = current_dir.parent
        
        env_path = current_dir / ".env"
        return cls(_env_file=str(env_path) if env_path.exists() else None)
