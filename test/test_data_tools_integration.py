"""Unit tests for data_tools package focusing on document processing."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.data_tools.loader_utils import DocumentLoader, DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    """Test document processing functionality without embeddings."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test documents
        self.doc_path = os.path.join(self.test_dir, "test.txt")
        with open(self.doc_path, "w") as f:
            f.write("Test document content.\nWith multiple lines.\n")
            
        # Create invalid file
        self.invalid_path = os.path.join(self.test_dir, "invalid.xyz")
        with open(self.invalid_path, "w") as f:
            f.write("Invalid file type")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_document_processing(self):
        """Test basic document processing without embeddings."""
        processor = DocumentProcessor()
        chunks = list(processor.process_document(self.doc_path))
        
        self.assertTrue(len(chunks) > 0, "Should generate chunks")
        self.assertIn("text", chunks[0], "Chunk should contain text")
        self.assertIn("metadata", chunks[0], "Chunk should contain metadata")
    
    def test_invalid_file_handling(self):
        """Test handling of invalid file types."""
        processor = DocumentProcessor()
        chunks = list(processor.process_document(self.invalid_path))
        self.assertEqual(len(chunks), 0, "Invalid file should produce no chunks")

class TestDocumentLoader(unittest.TestCase):
    """Test document loader with mocked embeddings."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test.db")
        
        # Create a test document
        self.doc_path = os.path.join(self.test_dir, "test.txt")
        with open(self.doc_path, "w") as f:
            f.write("Test document for loader")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    @patch('src.data_tools.loader_utils.Embeddings')
    def test_loader_interface(self, mock_embeddings):
        """Test loader interface with mocked embeddings."""
        # Setup mock
        mock_instance = Mock()
        mock_embeddings.return_value = mock_instance
        mock_instance.search.return_value = [{"text": "Test result", "score": 0.95}]
        
        # Test loader
        loader = DocumentLoader(self.db_path)
        chunks = list(loader.process_documents(self.doc_path))
        self.assertTrue(len(chunks) > 0, "Should generate chunks")
        
        # Test loading to database
        loader.load_to_database(chunks)
        mock_instance.add.assert_called_once()
        
        # Test search
        results = loader.search("test query")
        self.assertEqual(len(results), 1, "Should return mocked result")
        mock_instance.search.assert_called_once_with("test query", limit=5)

if __name__ == '__main__':
    unittest.main()
