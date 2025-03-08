"""Global state for txtai MCP server."""
from typing import Optional, Dict, Any, TYPE_CHECKING
from txtai.app import Application

if TYPE_CHECKING:
    from ..tools.causal_config import CausalBoostConfig

# Global Application instance
_txtai_app: Optional[Application] = None

# Global document cache to persist between requests
_document_cache: Dict[Any, str] = {}

# Global causal boost configuration
_causal_config: Optional[Any] = None

def get_txtai_app() -> Application:
    """Get the global txtai Application instance."""
    if _txtai_app is None:
        raise RuntimeError("TxtAI application not initialized")
    return _txtai_app

def set_txtai_app(app: Application) -> None:
    """Set the global txtai Application instance."""
    global _txtai_app
    _txtai_app = app

def get_document_cache() -> Dict[Any, str]:
    """Get the global document cache."""
    global _document_cache
    return _document_cache

def add_to_document_cache(doc_id: Any, text: str) -> None:
    """Add a document to the global cache."""
    global _document_cache
    _document_cache[doc_id] = text

def get_from_document_cache(doc_id: Any) -> Optional[str]:
    """Get a document from the global cache."""
    global _document_cache
    return _document_cache.get(doc_id)

# Aliases for backward compatibility
def add_document_to_cache(doc_id: str, text: str) -> None:
    """Alias for add_to_document_cache for backward compatibility."""
    add_to_document_cache(doc_id, text)

def get_document_from_cache(doc_id: Any) -> Optional[str]:
    """Alias for get_from_document_cache for backward compatibility."""
    return get_from_document_cache(doc_id)

def get_causal_config() -> Optional[Any]:
    """Get the global causal boost configuration.
    
    Returns:
        Optional[CausalBoostConfig]: The current causal boost configuration,
            or None if causal boost is not enabled.
    """
    global _causal_config
    return _causal_config

def set_causal_config(config: Optional[Any]) -> None:
    """Set the global causal boost configuration.
    
    Args:
        config: The causal boost configuration to use, or None to disable causal boost.
    """
    # Verify the config is a CausalBoostConfig instance
    if config is not None:
        from ..tools.causal_config import CausalBoostConfig
        if not isinstance(config, CausalBoostConfig):
            raise TypeError(f"Expected CausalBoostConfig, got {type(config)}")
    
    global _causal_config
    _causal_config = config
