"""Utility functions for document loading and processing."""

import os
import logging
from typing import List, Dict, Generator, Optional, Tuple, Union, Callable
from pathlib import Path
import uuid
import json
from datetime import datetime

from txtai.pipeline import Textractor
from txtai.embeddings import Embeddings

from .config import ProcessorConfig, EmbeddingsConfig, LoaderConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and text extraction."""

    def __init__(self, config: Optional[ProcessorConfig] = None):
        """Initialize the document processor.
        
        Args:
            config: Optional ProcessorConfig instance
        """
        self.config = config or ProcessorConfig()
        self.textractor = Textractor()

    def is_supported_file(self, filepath: str) -> bool:
        """Check if the file type is supported.
        
        Args:
            filepath: Path to the file
            
        Returns:
            bool: True if file type is supported
        """
        return Path(filepath).suffix.lower() in self.config.supported_extensions

    def extract_text(self, filepath: str) -> str:
        """Extract text content from a document.
        
        Args:
            filepath: Path to the document
            
        Returns:
            str: Extracted text content
        """
        return self.textractor(filepath)

    def process_text(self, text: str, source: str) -> Generator[Dict[str, str], None, None]:
        """Process text content into chunks with metadata.

        Args:
            text: Text content to process
            source: Source identifier (e.g. file path)

        Yields:
            Dict containing text chunk and metadata
        """
        # Split text into sections and paragraphs
        sections = text.split('\n## ')
        if len(sections) == 1:
            sections = text.split('\n# ')
            
        current_chunk = []
        current_length = 0
        
        for i, section in enumerate(sections):
            if i > 0:  # Add back the header marker except for first section
                section = '## ' + section
            
            # Split section into paragraphs
            paragraphs = [p.strip() for p in section.split('\n\n') if p.strip()]
            
            for para in paragraphs:
                para_length = len(para)
                
                # If adding this paragraph would exceed target length and we have enough content
                if current_length + para_length > self.config.chunk_size and current_length >= self.config.min_chunk_size:
                    yield self._create_chunk('\n\n'.join(current_chunk), source)
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(para)
                current_length += para_length
                
                # If this paragraph alone exceeds target length, split it
                if para_length > self.config.chunk_size:
                    sentences = [s.strip() + '.' for s in para.split('.') if s.strip()]
                    current_sentence_chunk = []
                    current_sentence_length = 0
                    
                    for sentence in sentences:
                        sentence_length = len(sentence)
                        if current_sentence_length + sentence_length > self.config.chunk_size:
                            if current_sentence_chunk:
                                yield self._create_chunk(' '.join(current_sentence_chunk), source)
                            current_sentence_chunk = [sentence]
                            current_sentence_length = sentence_length
                        else:
                            current_sentence_chunk.append(sentence)
                            current_sentence_length += sentence_length
                    
                    if current_sentence_chunk:
                        yield self._create_chunk(' '.join(current_sentence_chunk), source)
                    current_chunk = []
                    current_length = 0
        
        # Don't forget the last chunk
        if current_chunk and current_length >= self.config.min_chunk_size:
            yield self._create_chunk('\n\n'.join(current_chunk), source)

    def _create_chunk(self, text: str, source: str) -> Dict[str, str]:
        """Create a chunk with metadata.
        
        Args:
            text: Chunk text content
            source: Source identifier
            
        Returns:
            Dict with text and metadata
        """
        chunk_id = str(uuid.uuid4())
        return {
            "id": chunk_id,
            "text": text,
            "metadata": {
                "source": source,
                "chunk_id": chunk_id,
                "length": len(text),
                "created_at": datetime.utcnow().isoformat()
            }
        }

class DocumentLoader:
    """Handles loading documents into txtai database."""

    def __init__(self, config: Union[LoaderConfig, str]):
        """Initialize the document loader.
        
        Args:
            config: LoaderConfig instance or database URL string
        """
        if isinstance(config, str):
            self.config = LoaderConfig(db_url=config)
        else:
            self.config = config
            
        self.processor = DocumentProcessor(self.config.processor)
        self.embeddings = self._initialize_embeddings()
        self._total_processed = 0

    def _initialize_embeddings(self) -> Embeddings:
        """Initialize txtai embeddings with configuration."""
        # Get embeddings config
        config = self.config.embeddings.to_dict()
        
        # Add database configuration
        if isinstance(self.config.database.content, str) and self.config.database.content != "sqlite":
            # Using a client database or connection URL
            config["content"] = self.config.database.content
            if self.config.database.schema:
                config["schema"] = self.config.database.schema
        else:
            # Using sqlite
            config["content"] = True
            if self.config.database.sqlite_wal:
                config["sqlite"] = {"wal": True}
        
        # Set index path and writable status if provided
        if self.config.database.path:
            config["path"] = self.config.database.path
            config["writable"] = self.config.database.writable
        
        return Embeddings(config)

    def process_documents(self, input_path: Union[str, Path], recursive: bool = False) -> Generator[Dict[str, str], None, None]:
        """Process documents from a file or directory.
        
        Args:
            input_path: Path to file or directory to process
            recursive: If True and input_path is directory, process subdirectories
            
        Yields:
            Dict containing text chunk and metadata
        """
        input_path = Path(input_path)
        
        if input_path.is_file():
            if self.processor.is_supported_file(str(input_path)):
                text = self.processor.extract_text(str(input_path))
                yield from self.processor.process_text(text, str(input_path))
        elif input_path.is_dir():
            pattern = '**/*' if recursive else '*'
            for file_path in input_path.glob(pattern):
                if file_path.is_file() and self.processor.is_supported_file(str(file_path)):
                    try:
                        text = self.processor.extract_text(str(file_path))
                        yield from self.processor.process_text(text, str(file_path))
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {str(e)}")

    def load_to_database(self, chunks: Generator[Dict[str, str], None, None], 
                        progress_callback: Optional[Callable[[int], None]] = None) -> None:
        """Load processed chunks into txtai database.
        
        Args:
            chunks: Generator of processed text chunks
            progress_callback: Optional callback function to report progress
        """
        batch = []
        
        for chunk in chunks:
            batch.append((chunk["id"], chunk["text"], chunk["metadata"]))
            
            if len(batch) >= self.config.batch_size:
                self._process_batch(batch)
                self._total_processed += len(batch)
                
                if progress_callback and self._total_processed % self.config.progress_interval == 0:
                    progress_callback(self._total_processed)
                    
                batch = []
        
        # Process remaining chunks
        if batch:
            self._process_batch(batch)
            self._total_processed += len(batch)
            if progress_callback:
                progress_callback(self._total_processed)

    def _process_batch(self, batch: List[Tuple[str, str, dict]]):
        """Process a batch of documents.
        
        Args:
            batch: List of (id, text, metadata) tuples
        """
        self.embeddings.add(batch)

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for documents using semantic search.
        
        Args:
            query: Search query
            limit: Number of results to return
            
        Returns:
            List of search results
        """
        return self.embeddings.search(query, limit=limit)

    def delete_documents(self, ids: List[str]) -> None:
        """Delete documents from the database.
        
        Args:
            ids: List of document IDs to delete
        """
        self.embeddings.delete(ids)

    def update_documents(self, chunks: Generator[Dict[str, str], None, None],
                        progress_callback: Optional[Callable[[int], None]] = None) -> None:
        """Update existing documents in the database.
        
        This will upsert documents - add new ones and update existing ones.
        
        Args:
            chunks: Generator of processed text chunks
            progress_callback: Optional callback function to report progress
        """
        self.load_to_database(chunks, progress_callback)

    def get_document_count(self) -> int:
        """Get total number of documents in database.
        
        Returns:
            int: Number of documents
        """
        return len(self.embeddings)

    def get_stats(self) -> Dict:
        """Get database statistics.
        
        Returns:
            Dict containing database statistics
        """
        return {
            "total_documents": self.get_document_count(),
            "total_processed": self._total_processed,
            "config": {
                "model": self.config.embeddings.model_name,
                "batch_size": self.config.batch_size,
                "chunk_size": self.config.processor.chunk_size,
                "backend": self.config.embeddings.backend
            }
        }
