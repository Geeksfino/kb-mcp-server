"""Configuration for txtai MCP server."""
import os
from pathlib import Path
from typing import Dict, Any, Optional, Literal

from pydantic import validator
from pydantic_settings import BaseSettings
from txtai.app import Application


class TxtAISettings(BaseSettings):
    """Settings for txtai MCP server.
    
    Two configuration methods are supported:
    1. YAML config (recommended) - Use TXTAI_YAML_CONFIG to specify path
    2. Environment variables (fallback) - Use TXTAI_ prefixed variables
    """
    
    # YAML config path (optional)
    yaml_config: Optional[str] = None
    
    # Basic settings (fallback if no YAML)
    model_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    storage_mode: Literal["memory", "persistence"] = "memory"
    index_path: str = "~/.txtai/embeddings"
    
    @validator("yaml_config", "index_path")
    def expand_path(cls, v: Optional[str]) -> Optional[str]:
        """Expand user path if present."""
        return str(Path(v).expanduser()) if v else v
    
    @validator("storage_mode")
    def validate_storage(cls, v: str) -> str:
        """Validate storage mode."""
        if v not in ["memory", "persistence"]:
            raise ValueError("storage_mode must be 'memory' or 'persistence'")
        return v
    
    @classmethod
    def load(cls) -> "TxtAISettings":
        """Load settings from environment and .env file."""
        return super().load()
    
    def create_application(self) -> Application:
        """Create txtai Application instance.
        
        Returns:
            Application: txtai Application instance configured either through
                        YAML or environment variables.
        
        Raises:
            ValueError: If YAML config file not found
        """
        if self.yaml_config:
            # Use txtai's native YAML config
            yaml_path = Path(self.yaml_config)
            if not yaml_path.exists():
                raise ValueError(f"YAML config file not found: {yaml_path}")
            return Application(str(yaml_path))
        
        # Fallback to basic config
        config = {
            "writable": True,
            "embeddings": {
                "path": self.model_path,
                "content": True
            }
        }
        
        # Add persistence path if needed
        if self.storage_mode == "persistence":
            config["path"] = self.index_path
            
        return Application(config)
    
    class Config:
        """Pydantic config."""
        env_prefix = "TXTAI_"  # Use TXTAI_ prefix for all env vars
        env_file = ".env"
        env_file_encoding = "utf-8"
