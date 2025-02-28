#!/usr/bin/env python3
"""
Integration test for data_tools CLI functionality.

This script tests the command-line interface for document ingestion and knowledge graph generation.
It validates:
1. Document ingestion from files
2. Knowledge graph building
3. Search functionality (both standard and graph-enhanced)

Usage:
    python -m test.test_data_tools_cli
"""

import os
import sys
import tempfile
import unittest
import shutil
import logging
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_tools.config import save_config, create_default_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestDataToolsCLI(unittest.TestCase):
    """Test case for data_tools CLI functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create temporary directories
        cls.test_dir = tempfile.mkdtemp()
        cls.config_dir = tempfile.mkdtemp()
        
        # Create test configuration
        cls.config_path = os.path.join(cls.config_dir, "config.yaml")
        config = create_default_config()
        
        # Update paths in config to use temporary directory
        config["path"] = os.path.join(cls.test_dir, "knowledge-base")
        config["embeddings"]["storagepath"] = os.path.join(cls.test_dir, "embeddings")
        
        # Save configuration
        save_config(config, cls.config_path)
        logger.info(f"Created test configuration at {cls.config_path}")
        
        # Create test document
        cls.test_doc_path = os.path.join(cls.test_dir, "test_document.txt")
        with open(cls.test_doc_path, "w") as f:
            f.write("""# Machine Learning Overview

Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.
It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns,
and make decisions with minimal human intervention.

There are three main types of machine learning algorithms:
1. Supervised Learning: The algorithm is trained on labeled data, learning to map input to output.
2. Unsupervised Learning: The algorithm finds patterns in unlabeled data.
3. Reinforcement Learning: The algorithm learns by interacting with an environment and receiving rewards or penalties.
""")
        logger.info(f"Created test document at {cls.test_doc_path}")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests are done."""
        # Remove temporary directories
        shutil.rmtree(cls.test_dir)
        shutil.rmtree(cls.config_dir)
        logger.info("Removed temporary test directories")

    def run_cli_command(self, args):
        """Run a CLI command as a subprocess and return the result."""
        cmd = [sys.executable, "-m", "src.data_tools.cli"] + args
        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}")
            logger.error(f"STDERR: {e.stderr}")
            return False, e.stderr

    def test_1_document_ingestion(self):
        """Test document ingestion functionality."""
        logger.info("Testing document ingestion...")
        
        # Run CLI with ingest command
        success, output = self.run_cli_command([
            "--config", self.config_path,
            "ingest",
            "file",
            self.test_doc_path
        ])
        
        self.assertTrue(success, f"Document ingestion failed: {output}")
        logger.info("Document ingestion test passed")

    def test_2_graph_building(self):
        """Test knowledge graph building functionality."""
        logger.info("Testing knowledge graph building...")
        
        # Run CLI with graph command
        success, output = self.run_cli_command([
            "--config", self.config_path,
            "graph",
            "--min-similarity", "0.6"
        ])
        
        self.assertTrue(success, f"Knowledge graph building failed: {output}")
        logger.info("Knowledge graph building test passed")

    def test_3_search(self):
        """Test search functionality."""
        logger.info("Testing standard search...")
        
        # Run CLI with search command
        success, output = self.run_cli_command([
            "--config", self.config_path,
            "search",
            "What is machine learning?",
            "--limit", "3"
        ])
        
        self.assertTrue(success, f"Standard search failed: {output}")
        logger.info("Standard search test passed")

    def test_4_graph_search(self):
        """Test search with knowledge graph functionality."""
        logger.info("Testing search with knowledge graph...")
        
        # Run CLI with search command using graph
        success, output = self.run_cli_command([
            "--config", self.config_path,
            "search",
            "What is machine learning?",
            "--limit", "3",
            "--use-graph"
        ])
        
        self.assertTrue(success, f"Graph search failed: {output}")
        logger.info("Graph search test passed")


if __name__ == "__main__":
    unittest.main()
