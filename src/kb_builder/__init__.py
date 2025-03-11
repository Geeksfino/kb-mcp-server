"""Knowledge Base Builder - Tools for knowledge base construction and search.

This package provides tools for building and querying knowledge bases using txtai.
"""

from .cli import main, create_application, setup_logging
from .settings import Settings

__all__ = ["main", "Settings", "create_application", "setup_logging"]
