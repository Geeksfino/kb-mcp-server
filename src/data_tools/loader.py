#!/usr/bin/env python3
"""
Document loader for ingesting various file formats into txtai.

This module provides a DocumentLoader class that can process PDF, text, Markdown,
HTML, and Word documents using txtai's Textractor pipeline.
"""

import os
import logging
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Iterator, Any, Set

from txtai.pipeline import Textractor
from txtai.app import Application

logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    Document loader for ingesting various file formats into txtai.
    
    This class handles the extraction of text from documents using txtai's
    Textractor pipeline, which leverages Apache Tika for comprehensive
    document processing.
    """
    
    # File extensions supported by direct text extraction (simple formats)
    SIMPLE_FORMATS = {'.txt', '.md', '.markdown', '.html', '.htm'}
    
    # File extensions that require Textractor/Tika (complex formats)
    COMPLEX_FORMATS = {'.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'}
    
    def __init__(self, app: Optional[Application] = None):
        """
        Initialize the DocumentLoader.
        
        Args:
            app: Optional txtai Application instance. If provided, will use app's configuration.
        """
        self.app = app
        
        # Extract processor configuration from app config if available
        self.processor_config = {}
        if app:
            self.processor_config = app.config.get("processor", {})
        
        # Set up segmentation parameters
        self.min_length = self.processor_config.get("minlength", 100)
        self.max_length = self.processor_config.get("maxlength", 1000)
        self.segmentation = self.processor_config.get("segmentation", "paragraphs")
        
        # Set up chunking parameters
        self.chunk_size = self.processor_config.get("chunk_size", 1000)
        self.chunk_overlap = self.processor_config.get("chunk_overlap", 100)
        
        # Initialize Textractor with configuration
        textractor_params = {
            "minlength": self.min_length,
            "cleantext": self.processor_config.get("cleantext", True)
        }
        
        # Configure segmentation strategy
        if self.segmentation == "sentences":
            textractor_params["sentences"] = True
        elif self.segmentation == "paragraphs":
            textractor_params["paragraphs"] = True
        elif self.segmentation == "sections":
            textractor_params["sections"] = True
            
        # Create Textractor instance
        self.textractor = Textractor(**textractor_params)
        
    def process_file(self, file_path: Union[str, Path], bypass_textractor: bool = False) -> List[Dict[str, Any]]:
        """
        Process a file and extract text segments.
        
        Args:
            file_path: Path to the file
            bypass_textractor: If True, use direct text extraction for simple formats
            
        Returns:
            List of document dictionaries with text and metadata
        """
        file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
        
        # Generate a document ID based on file path
        doc_id = self._generate_doc_id(file_path)
        
        # Extract text from the file
        segments = self._extract_text(file_path, bypass_textractor)
        
        if not segments:
            logger.warning(f"No text extracted from {file_path}")
            return []
        
        logger.info(f"Extracted {len(segments)} segments from {file_path}")
        
        # Create document dictionaries with metadata
        documents = []
        for i, segment in enumerate(segments):
            # Skip empty segments
            if not segment.strip():
                continue
                
            # Extract metadata
            metadata = self._create_metadata(file_path, i, len(segments))
            
            # Create document in the format expected by txtai: (id, content, tags)
            # Where content can be a string or a dict with text field
            document_id = f"{doc_id}-{i}"
            document_content = {
                "text": segment,
                "metadata": metadata  # Store metadata directly, not as JSON string
            }
            
            # Add as a tuple (id, content, tags) where tags is None
            documents.append((document_id, document_content, None))
            
        return documents
    
    def process_directory(self, directory_path: Union[str, Path], 
                         recursive: bool = True,
                         extensions: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Process all files in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to process subdirectories
            extensions: Set of file extensions to process (e.g., {'.pdf', '.txt'})
            
        Returns:
            List of document dictionaries with text and metadata
            
        Raises:
            NotADirectoryError: If the path is not a directory
        """
        directory_path = Path(directory_path) if not isinstance(directory_path, Path) else directory_path
        
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        logger.info(f"Processing directory: {directory_path}, recursive: {recursive}")
        
        # If no extensions specified, use all supported formats
        if extensions is None:
            extensions = self.SIMPLE_FORMATS.union(self.COMPLEX_FORMATS)
        
        # Convert extensions to lowercase for case-insensitive matching
        extensions = {ext.lower() for ext in extensions}
        
        # Process files
        documents = []
        for file_path in self._find_files(directory_path, extensions, recursive):
            try:
                file_documents = self.process_file(file_path)
                documents.extend(file_documents)
                logger.info(f"Added {len(file_documents)} segments from {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                
        return documents
    
    def _extract_text(self, file_path: Path, bypass_textractor: bool = False) -> List[str]:
        """
        Extract text from a file using the appropriate method.
        
        Args:
            file_path: Path to the file
            bypass_textractor: If True, use direct text extraction for simple formats
            
        Returns:
            List of text segments
        """
        # Determine if we can use direct text extraction
        if file_path.suffix.lower() in self.SIMPLE_FORMATS and bypass_textractor:
            logger.info(f"Using direct text extraction for {file_path}")
            return self._extract_text_directly(file_path)
        else:
            # Use Textractor for complex formats
            logger.info(f"Using Textractor for {file_path}")
            return self.textractor(str(file_path))
    
    def _extract_text_directly(self, file_path: Path) -> List[str]:
        """
        Extract text directly from simple file formats without using Textractor.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of text segments
        """
        try:
            # Try UTF-8 encoding first
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fall back to latin-1 if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return []
        
        # Segment the content based on the configured strategy
        if self.segmentation == "sentences":
            # Simple sentence splitting (not as robust as Textractor)
            import re
            segments = []
            for paragraph in content.split('\n\n'):
                if not paragraph.strip():
                    continue
                # Split paragraph into sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                segments.extend([s for s in sentences if len(s) >= self.min_length])
        else:  # Default to paragraphs
            # Split by double newlines for paragraphs
            segments = [p.strip() for p in content.split('\n\n') 
                       if len(p.strip()) >= self.min_length]
        
        # Apply length constraints
        segments = [s[:self.max_length] for s in segments if len(s) >= self.min_length]
        
        return segments
    
    def _create_metadata(self, file_path: Path, segment_index: int, total_segments: int) -> Dict[str, Any]:
        """
        Create metadata for a document segment.
        
        Args:
            file_path: Path to the source file
            segment_index: Index of the segment within the file
            total_segments: Total number of segments in the file
            
        Returns:
            Dictionary of metadata
        """
        # Get file stats
        stats = file_path.stat()
        
        return {
            "source": str(file_path),
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "size_bytes": stats.st_size,
            "modified_time": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "segment_index": segment_index,
            "total_segments": total_segments,
            "extraction_time": datetime.now().isoformat()
        }
    
    def _find_files(self, directory: Path, extensions: Set[str], recursive: bool) -> Iterator[Path]:
        """
        Find files with the specified extensions in the directory.
        
        Args:
            directory: Directory to search
            extensions: Set of file extensions to include
            recursive: Whether to search subdirectories
            
        Yields:
            Paths to matching files
        """
        # Use recursive glob if recursive is True, otherwise just list directory
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                yield file_path
                
    def _generate_doc_id(self, file_path: Path) -> str:
        """
        Generate a unique document ID based on the file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document ID as a string
        """
        return hashlib.md5(str(file_path).encode()).hexdigest()
