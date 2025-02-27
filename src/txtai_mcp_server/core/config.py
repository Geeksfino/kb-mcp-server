"""Configuration for txtai MCP server."""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal, Union, Tuple

from pydantic import validator
from pydantic_settings import BaseSettings
from txtai.app import Application

logger = logging.getLogger(__name__)

class TxtAISettings(BaseSettings):
    """Settings for txtai MCP server.
    
    Three configuration methods are supported:
    1. Embeddings path - Use a pre-built embeddings directory or archive file
    2. YAML config - Use a YAML configuration file
    3. Environment variables - Use TXTAI_ prefixed variables
    """
    
    # Configuration path (can be embeddings path or YAML config)
    yaml_config: Optional[str] = None
    
    # Basic settings (fallback if no yaml_config)
    model_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_path: str = "~/.txtai/embeddings"
    
    # Model settings
    model_gpu: bool = True
    model_normalize: bool = True
    store_content: bool = True
    
    @validator("yaml_config", "index_path")
    def expand_path(cls, v: Optional[str]) -> Optional[str]:
        """Expand user path if present."""
        return str(Path(v).expanduser()) if v else v
    
    @classmethod
    def load(cls) -> "TxtAISettings":
        """Load settings from environment and .env file."""
        return cls.model_validate({})  # Empty dict will load from env vars
    
    def create_application(self) -> Application:
        """Create txtai Application instance.
        
        This method creates a txtai Application instance based on the configuration.
        It supports three modes of operation:
        
        1. If yaml_config points to an embeddings directory or archive file,
           it will load the embeddings directly using txtai's built-in functionality.
           
        2. If yaml_config points to a YAML configuration file, it will pass
           the path directly to the Application constructor.
           
        3. If no yaml_config is provided, it will create an Application using the
           settings from environment variables or defaults.
        
        Returns:
            Application: txtai Application instance configured appropriately.
        """
        if self.yaml_config:
            # Let txtai handle the path - it can automatically determine if it's
            # an embeddings directory, archive, or YAML config
            logger.info(f"Creating Application from path: {self.yaml_config}")
            return Application(self.yaml_config)
        
        # Otherwise configure through settings
        config = {
            "path": self.model_path,
            "content": self.store_content,
            "writable": True,  # Set writable at root level
            "embeddings": {
                "path": self.model_path,
                "storagepath": self.index_path,
                "gpu": self.model_gpu,
                "normalize": self.model_normalize,
                "writable": True  # Also set writable in embeddings
            }
        }
        logger.debug(f"Creating Application with default configuration")
        return Application(config)
        
    @classmethod
    def from_embeddings(cls, embeddings_path: str) -> Tuple["TxtAISettings", Application]:
        """Create settings and application directly from embeddings.
        
        This is a convenience method that creates a TxtAISettings instance
        and initializes an Application from an embeddings path in one step.
        
        Args:
            embeddings_path: Path to embeddings directory or archive file
            
        Returns:
            Tuple of (TxtAISettings, Application)
        """
        # Create settings with the embeddings path as yaml_config
        settings = cls(yaml_config=embeddings_path)
        
        # Create and return application
        app = settings.create_application()
        
        logger.info(f"Successfully loaded embeddings from: {embeddings_path}")
        return settings, app

    class Config:
        env_prefix = "TXTAI_"  # Look for TXTAI_ prefixed env vars
        extra = "allow"  # Allow extra fields from env vars
        env_file = ".env"
        env_file_encoding = "utf-8"
