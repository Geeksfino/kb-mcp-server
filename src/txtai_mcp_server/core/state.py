"""Global state for txtai MCP server."""
from typing import Optional, Dict, Any
from txtai.app import Application

# Global Application instance
_txtai_app: Optional[Application] = None

# Global document cache to persist between requests
_document_cache: Dict[Any, str] = {}

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

def add_document_to_cache(doc_id: str, text: str) -> None:
    """Add document text to cache."""
    global _document_cache
    _document_cache[doc_id] = text

def get_document_from_cache(doc_id: Any) -> Optional[str]:
    """Alias for get_from_document_cache for backward compatibility."""
    return get_from_document_cache(doc_id)
