"""
Context objects for the txtai MCP server.
"""
from dataclasses import dataclass
from typing import Optional

from txtai.embeddings import Embeddings


@dataclass
class TxtAIContext:
    """Context for txtai components."""
    embeddings: Embeddings
