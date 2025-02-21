"""Resources for the txtai MCP server."""

from .config import register_config_resources
from .models import register_model_resources

__all__ = ["register_config_resources", "register_model_resources"]
