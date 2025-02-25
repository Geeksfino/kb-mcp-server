"""
Context objects for the txtai MCP server.
"""
import logging
from dataclasses import dataclass
from txtai.app import Application

logger = logging.getLogger(__name__)

@dataclass
class TxtAIContext:
    """Context object for txtai application."""
    app: Application

    def __post_init__(self):
        """Log initialization."""
        logger.debug("TxtAIContext initialized with Application")
