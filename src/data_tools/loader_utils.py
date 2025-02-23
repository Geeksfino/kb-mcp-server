"""Document loader utilities."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Generator, Tuple
import json

from txtai.embeddings import Embeddings
from datasets import load_dataset
from txtai_mcp_server.core.config import TxtAISettings

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Load and process documents into txtai database."""
    
    def __init__(self, config=None):
        """Initialize loader with configuration."""
        # Use MCP server settings if no config provided
        if config is None:
            settings = TxtAISettings.load()
            self.config = settings.get_embeddings_config()
        else:
            self.config = config
            
        logger.info(f"Initializing embeddings with config: {self.config}")
        self.embeddings = Embeddings(self.config)
        
        # Log database paths
        db_path = Path(self.config.database.path).expanduser()
        logger.info(f"Initializing embeddings database at: {db_path}")
        
        # Check if database files exist
        content_db = db_path / "content.db"
        vectors_db = db_path / "vectors.faiss"
        logger.info(f"Content database exists: {content_db.exists()}")
        logger.info(f"Vectors database exists: {vectors_db.exists()}")
        
    def process_huggingface_dataset(self, dataset_name: str, split: str, text_field: Optional[str] = None) -> None:
        """Process documents from a HuggingFace dataset.
        
        Args:
            dataset_name: Name of the dataset
            split: Dataset split to load
            text_field: Field to use as text content. If not provided, will try to auto-detect.
        """
        logger.info(f"Loading dataset {dataset_name} (split: {split})")
        
        # Load dataset
        dataset = load_dataset(dataset_name, split=split)
        logger.info(f"Dataset loaded successfully with {len(dataset)} items")
        
        # Auto-detect text field if not provided
        if not text_field:
            # Look for common text field names
            common_fields = ["text", "content", "question", "answer", "title", "body"]
            for field in common_fields:
                if field in dataset.features:
                    text_field = field
                    break
                    
            if not text_field:
                raise ValueError(f"Could not auto-detect text field. Please specify using --text-field. Available fields: {list(dataset.features.keys())}")
                
        logger.info(f"Auto-detected text field: {text_field}")
        
        # Stream data to embeddings index
        def stream():
            for idx, item in enumerate(dataset):
                text = item[text_field]
                if text and isinstance(text, str):
                    yield (idx, text, None)
                    
        # Index the data
        self.embeddings.index(stream())
        logger.info(f"Dataset processing complete: {len(dataset)} items processed")
        
        # Save embeddings to disk
        logger.info("Saving embeddings to disk...")
        self.embeddings.save(self.config.database.path)
        
        # Verify files were created
        content_db = Path(self.config.database.path).expanduser() / "content.db"
        vectors_db = Path(self.config.database.path).expanduser() / "vectors.faiss"
        logger.info(f"Content database exists: {content_db.exists()}")
        logger.info(f"Vectors database exists: {vectors_db.exists()}")
        
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for documents using semantic search.
        
        Args:
            query: Search query
            limit: Number of results to return
            
        Returns:
            List of search results, each containing text and score
        """
        # Get raw search results
        results = self.embeddings.search(query, limit=limit)
        logger.info(f"Raw search results: {results}")
        
        # Format results
        formatted_results = []
        for result in results:
            if isinstance(result, dict):
                # Handle case where result is already a dict
                formatted_results.append({
                    'text': result.get('text', ''),
                    'score': result.get('score', 0.0)
                })
            elif isinstance(result, (list, tuple)) and len(result) >= 2:
                # Handle case where result is (text, score) tuple
                formatted_results.append({
                    'text': result[0],
                    'score': result[1]
                })
                
        return formatted_results
