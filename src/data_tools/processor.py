#!/usr/bin/env python3
"""
Document processor for converting extracted text into embeddings.

This module provides a DocumentProcessor class that converts the text extracted
by the DocumentLoader into embeddings for similarity search and BM25 indexing.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Union, Any, Iterator, Tuple

from txtai.app import Application

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Document processor for converting extracted text into embeddings.
    
    This class handles the conversion of extracted text into embeddings
    and indexes them for similarity search and BM25 (sparse) indexing.
    """
    
    def __init__(self, app: Application):
        """
        Initialize the DocumentProcessor with a txtai Application.
        
        Args:
            app: A configured txtai Application instance
        """
        self.app = app
        
        # Check if hybrid search is enabled
        self.hybrid_enabled = self.app.config.get("embeddings", {}).get("hybrid", False)
        if self.hybrid_enabled:
            logger.info("Hybrid search (dense + sparse) is enabled")
        else:
            logger.info("Only dense embeddings will be used (hybrid search disabled)")
    
    def process_documents(self, documents: List[Union[Dict[str, Any], Tuple]]) -> int:
        """
        Process documents and add them to the index.
        
        Args:
            documents: List of document tuples (id, content, tags) or dictionaries
            
        Returns:
            Number of documents indexed
        """
        if not documents:
            logger.warning("No documents to process")
            return 0
        
        logger.info(f"Processing {len(documents)} documents")
        
        # Add documents to the index
        # Documents should already be in the format expected by txtai: (id, content, tags)
        self.app.add(documents)
        
        # Build the index
        self.app.index()
        
        logger.info(f"Indexed {len(documents)} documents")
        return len(documents)
    
    def save_embeddings(self, path: Optional[str] = None) -> str:
        """
        Save the embeddings to disk.
        
        Args:
            path: Path to save the embeddings (optional, uses config path if not provided)
            
        Returns:
            Path where embeddings were saved
        """
        # If path not provided, use the one from config
        if not path:
            path = self.app.config.get("embeddings", {}).get("storagepath")
            if not path:
                path = os.path.join(os.getcwd(), ".txtai", "embeddings")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        logger.info(f"Saving embeddings to {path}")
        
        # Access the embeddings object directly and save it
        self.app.embeddings.save(path)
        
        return path
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed documents.
        
        Returns:
            Dictionary with statistics
        """
        # Get the number of documents
        count = len(self.app.search("", limit=1))
        
        # Get index information if available
        index_info = {}
        if hasattr(self.app.embeddings, "index") and self.app.embeddings.index:
            index = self.app.embeddings.index
            index_info = {
                "index_type": type(index).__name__,
                "dimensions": getattr(index, "dimensions", None),
                "hybrid_enabled": self.hybrid_enabled
            }
        
        return {
            "document_count": count,
            "index_info": index_info,
            "config": {
                "model": self.app.config.get("embeddings", {}).get("path"),
                "storage_type": self.app.config.get("embeddings", {}).get("storagetype")
            }
        }
