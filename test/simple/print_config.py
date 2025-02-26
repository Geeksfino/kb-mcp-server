"""Script to print out txtai configuration."""
import logging
from txtai_mcp_server.core.config import TxtAISettings
from txtai.app import Application
import yaml

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_default_config():
    """Print the default configuration."""
    # Get default config
    settings = TxtAISettings.load()
    config = {
        "path": settings.model_path,
        "content": settings.store_content,
        "embeddings": {
            "path": settings.model_path,
            "storagetype": settings.storage_mode,
            "storagepath": settings.index_path,
            "gpu": settings.model_gpu,
            "normalize": settings.model_normalize
        }
    }
    print("\nDefault Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Create application and print its config
    app = Application(config)
    print("\nApplication Configuration:")
    print(yaml.dump(app.config, default_flow_style=False))

if __name__ == "__main__":
    print_default_config()