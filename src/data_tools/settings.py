"""
Settings module for data_tools.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

class Settings:
    """Settings for data_tools."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = {}
        
        # Load configuration if available
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        import json
        import yaml
        
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load based on file extension
        if path.suffix.lower() in ('.yml', '.yaml'):
            with open(path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")
        
        return self.config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value
        
    @property
    def data_dir(self) -> Path:
        """
        Get the data directory.
        
        Returns:
            Path to data directory
        """
        data_dir = self.get('data_dir', '.txtai')
        return Path(data_dir)
    
    @property
    def index_path(self) -> Path:
        """
        Get the index path.
        
        Returns:
            Path to index
        """
        return self.data_dir / 'index'
    
    @property
    def graph_path(self) -> Path:
        """
        Get the graph path.
        
        Returns:
            Path to graph
        """
        return self.data_dir / 'graph.pkl'
