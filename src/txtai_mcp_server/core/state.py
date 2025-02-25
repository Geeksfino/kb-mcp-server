"""Global state for txtai MCP server."""
from typing import Optional
from txtai.app import Application

# Global Application instance
_txtai_app: Optional[Application] = None

def get_txtai_app() -> Application:
    """Get the global txtai Application instance."""
    if _txtai_app is None:
        raise RuntimeError("TxtAI application not initialized")
    return _txtai_app

def set_txtai_app(app: Application) -> None:
    """Set the global txtai Application instance."""
    global _txtai_app
    _txtai_app = app
