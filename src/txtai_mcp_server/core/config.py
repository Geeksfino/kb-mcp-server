"""Configuration for txtai MCP server."""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal

from pydantic import validator
from pydantic_settings import BaseSettings
from txtai.app import Application

logger = logging.getLogger(__name__)

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
    
    # Model settings
    model_gpu: bool = True
    model_normalize: bool = True
    store_content: bool = True
    
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
        return cls.model_validate({})  # Empty dict will load from env vars
    
    def create_application(self) -> Application:
        """Create txtai Application instance.
        
        Returns:
            Application: txtai Application instance configured either through
                        YAML or environment variables.
        
        Raises:
            ValueError: If yaml_config is specified but file not found.
        """
        if self.yaml_config:
            yaml_path = Path(self.yaml_config)
            if not yaml_path.exists():
                raise ValueError(f"YAML config file not found: {yaml_path}")
            logger.debug(f"Loading configuration from YAML file: {yaml_path}")
            with open(yaml_path) as f:
                import yaml
                yaml_config = yaml.safe_load(f)
                logger.debug(f"Raw YAML configuration: {yaml_config}")
                
                # Set writable flag at both root level and embeddings
                yaml_config["writable"] = True
                if "embeddings" in yaml_config:
                    yaml_config["embeddings"]["writable"] = True
                
                logger.debug(f"Modified configuration: {yaml_config}")
                
                # Create temporary YAML with modified config
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as tmp:
                    yaml.dump(yaml_config, tmp)
                    tmp_path = tmp.name
                
                logger.debug(f"Creating Application with temporary YAML path: {tmp_path}")
                return Application(tmp_path)
        
        # Otherwise configure through settings
        config = {
            "path": self.model_path,
            "content": self.store_content,
            "writable": True,  # Set writable at root level
            "embeddings": {
                "path": self.model_path,
                "storagetype": self.storage_mode,
                "storagepath": self.index_path,
                "gpu": self.model_gpu,
                "normalize": self.model_normalize,
                "writable": True  # Also set writable in embeddings
            }
        }
        logger.debug(f"Using default configuration: {config}")
        return Application(config)

    class Config:
        env_prefix = "TXTAI_"  # Look for TXTAI_ prefixed env vars
        extra = "allow"  # Allow extra fields from env vars
        env_file = ".env"
        env_file_encoding = "utf-8"
