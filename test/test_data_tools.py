"""Tests for data_tools package."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from txtai.embeddings import Embeddings

from src.data_tools.loader_utils import DocumentProcessor, DocumentLoader


class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor(chunk_size=100, overlap=20)
        
    def test_is_supported_file(self):
        """Test file type support checking."""
        self.assertTrue(self.processor.is_supported_file("test.pdf"))
        self.assertTrue(self.processor.is_supported_file("test.txt"))
        self.assertTrue(self.processor.is_supported_file("test.md"))
        self.assertTrue(self.processor.is_supported_file("test.doc"))
        self.assertTrue(self.processor.is_supported_file("test.docx"))
        self.assertFalse(self.processor.is_supported_file("test.xyz"))
        
    def test_chunk_text(self):
        """Test text chunking functionality."""
        text = "This is a test sentence. This is another test sentence. " * 5
        chunks = self.processor.chunk_text(text)
        
        # Verify chunks are created
        self.assertTrue(len(chunks) > 0)
        
        # Verify chunk size
        for chunk in chunks[:-1]:  # All but last chunk
            self.assertLessEqual(len(chunk), self.processor.chunk_size)
            
        # Verify chunks end with complete sentences where possible
        for chunk in chunks[:-1]:
            self.assertTrue(chunk.endswith("."))
            
    @patch('src.data_tools.loader_utils.Textractor')
    def test_extract_text(self, mock_textractor):
        """Test text extraction from documents."""
        mock_instance = MagicMock()
        mock_instance.return_value = "Extracted text"
        mock_textractor.return_value = mock_instance
        
        result = self.processor.extract_text("test.pdf")
        self.assertEqual(result, "Extracted text")
        mock_instance.assert_called_once_with("test.pdf")


class TestDocumentLoader(unittest.TestCase):
    """Test cases for DocumentLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.db_url = "http://localhost:8000"
        self.loader = DocumentLoader(self.db_url)
        
    @patch('src.data_tools.loader_utils.Embeddings')
    def test_initialize_embeddings(self, mock_embeddings):
        """Test embeddings initialization."""
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance
        
        loader = DocumentLoader(self.db_url)
        self.assertIsNotNone(loader.embeddings)
        
        # Verify embeddings were initialized with correct config
        mock_embeddings.assert_called_once()
        config = mock_embeddings.call_args[0][0]
        self.assertEqual(config["path"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertTrue(config["content"])
        self.assertTrue(config["contentid"])
        self.assertTrue(config["gpu"])
        
    def test_process_documents_single_file(self):
        """Test processing a single document."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            # Write test content
            content = "Test content. More test content."
            tmp.write(content.encode())
            tmp.flush()
            
            # Process the file
            chunks = list(self.loader.process_documents(tmp.name))
            
            # Verify chunks were created
            self.assertTrue(len(chunks) > 0)
            for chunk in chunks:
                self.assertIn("text", chunk)
                self.assertIn("source", chunk)
                self.assertIn("chunk_id", chunk)
                self.assertIn("total_chunks", chunk)
                
    def test_process_documents_directory(self):
        """Test processing a directory of documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                with open(os.path.join(tmpdir, f"test{i}.txt"), "w") as f:
                    f.write(f"Test content {i}")
                    
            # Process the directory
            chunks = list(self.loader.process_documents(tmpdir))
            
            # Verify chunks were created for each file
            self.assertTrue(len(chunks) > 0)
            
            # Verify unique sources
            sources = {chunk["source"] for chunk in chunks}
            self.assertEqual(len(sources), 3)
            
    @patch('src.data_tools.loader_utils.Embeddings')
    def test_load_to_database(self, mock_embeddings):
        """Test loading documents into database."""
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance
        
        # Create test chunks
        chunks = [
            {
                "text": f"Test content {i}",
                "source": f"test{i}.txt",
                "chunk_id": i,
                "total_chunks": 3
            }
            for i in range(3)
        ]
        
        # Load chunks
        loader = DocumentLoader(self.db_url)
        loader.load_to_database((c for c in chunks), batch_size=2)
        
        # Verify upsert was called with correct data
        self.assertTrue(mock_instance.upsert.called)
        total_items = sum(len(call[0][0]) for call in mock_instance.upsert.call_args_list)
        self.assertEqual(total_items, 3)  # Total number of chunks


if __name__ == '__main__':
    unittest.main()
