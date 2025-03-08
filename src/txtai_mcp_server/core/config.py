"""Configuration for txtai MCP server."""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal, Union, Tuple

from pydantic import field_validator, ConfigDict
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
    
    # Direct embeddings path (highest priority)
    embeddings_path: Optional[str] = None
    
    # Basic settings (fallback if no yaml_config)
    model_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    index_path: str = "~/.txtai/embeddings"
    
    # Model settings
    model_gpu: bool = True
    model_normalize: bool = True
    store_content: bool = True
    
    @field_validator("yaml_config", "index_path", "embeddings_path")
    @classmethod
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
        
        1. If embeddings_path is provided, it will load the embeddings directly
           using txtai's built-in functionality. This has highest priority.
           
        2. If yaml_config points to an embeddings directory or archive file,
           it will load the embeddings directly using txtai's built-in functionality.
            
        3. If yaml_config points to a YAML configuration file, it will pass
           the path directly to the Application constructor.
            
        4. If no embeddings_path or yaml_config is provided, it will create an Application using the
           settings from environment variables or defaults.
        
        Returns:
            Application: txtai Application instance configured appropriately.
        """
        # Priority 1: Direct embeddings path
        if self.embeddings_path:
            logger.info(f"Creating Application from embeddings path: {self.embeddings_path}")
            # For embeddings path, we need to use a YAML string with the path
            if os.path.isfile(self.embeddings_path) and self.embeddings_path.endswith(('.tar.gz', '.tgz')):
                # For archive files, use a YAML string with the path
                logger.info(f"Loading embeddings from archive file: {self.embeddings_path}")
                return Application(f"path: {self.embeddings_path}")
            else:
                # For directories, load directly
                logger.info(f"Loading embeddings from directory: {self.embeddings_path}")
                return Application(f"path: {self.embeddings_path}")
        
        # Priority 2: YAML config path
        elif self.yaml_config:
            # Let txtai handle the path - it can automatically determine if it's
            # an embeddings directory, archive, or YAML config
            logger.info(f"Creating Application from path: {self.yaml_config}")
            return Application(self.yaml_config)
        
        # Priority 3: Configure through settings
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
        # Create settings with the embeddings path
        settings = cls(embeddings_path=embeddings_path)
        
        # Create and return application
        app = settings.create_application()
        
        logger.info(f"Successfully loaded embeddings from: {embeddings_path}")
        return settings, app
    
    model_config = ConfigDict(
        env_prefix="TXTAI_",  # Look for TXTAI_ prefixed env vars
        extra="allow",  # Allow extra fields from env vars
        env_file=".env",
        env_file_encoding="utf-8"
    )
