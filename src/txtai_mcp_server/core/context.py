"""
Context objects for the txtai MCP server.
"""
import logging
from dataclasses import dataclass
from txtai.app import Application

logger = logging.getLogger(__name__)

@dataclass
class TxtAIContext:
    """Context for txtai components.
    
    This class manages the txtai Application instance and provides access to:
    1. Core features (search, embeddings, extraction) via app
    2. Dynamic features (graph, LLM, NER) added on demand
    
    Example:
        ctx = TxtAIContext(app=Application(...))
        
        # Core features via app
        results = ctx.app.search(...)
        answers = ctx.app.extract(...)
        
        # Dynamic features (added when needed)
        if not hasattr(ctx, "graph"):
            ctx.graph = GraphFactory.create(...)
    """
    app: Application  # txtai Application instance

    def __post_init__(self):
        """Log initialization."""
        logger.debug("TxtAIContext initialized with Application")
