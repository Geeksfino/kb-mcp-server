"""Test package for txtai-mcp-server.

This package contains test modules for various txtai functionalities.
"""

# Version of the test package
__version__ = "0.1.0"

# List of modules to export when using 'from test import *'
__all__ = ['test_semantic_search', 'test_qa_database']

# You could also import and expose specific functions
from .test_semantic_search import main as run_semantic_search_test
from .test_qa_database import main as run_qa_database_test