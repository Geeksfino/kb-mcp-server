"""
Graph building module for knowledge graph construction.

This module implements different strategies for building knowledge graphs:
1. Semantic Graph Builder: Builds graphs based on semantic similarity
2. Entity Graph Builder: Builds graphs by extracting entities and relationships using LLMs
3. Hybrid Graph Builder: Combines semantic similarity and entity extraction approaches
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple

import networkx as nx
import numpy as np
from txtai.embeddings import Embeddings

logger = logging.getLogger(__name__)


class GraphBuilder(ABC):
    """Abstract base class for graph building strategies."""

    @abstractmethod
    def build(self, documents: List[Dict[str, Any]], **kwargs) -> nx.Graph:
        """
        Build a graph from documents.

        Args:
            documents: List of document dictionaries
            **kwargs: Additional arguments for graph building

        Returns:
            A NetworkX graph
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the graph to a file.

        Args:
            path: Path to save the graph
        """
        pass

    @abstractmethod
    def load(self, path: str) -> nx.Graph:
        """
        Load a graph from a file.

        Args:
            path: Path to load the graph from

        Returns:
            A NetworkX graph
        """
        pass


class SemanticGraphBuilder(GraphBuilder):
    """
    Build a semantic graph from documents based on similarity scores.
    """
    
    def __init__(self, embeddings, config=None):
        """
        Initialize the semantic graph builder.
        
        Args:
            embeddings: Embeddings instance
            config: Configuration dictionary
        """
        self.embeddings = embeddings
        self.config = config or {}
        self.threshold = self.config.get("threshold", 0.5)
        self.graph = None
        self.logger = logging.getLogger(__name__)
        
        # Check if txtai LLM is available
        try:
            import txtai.pipeline
            self.llm_available = True
        except ImportError:
            self.logger.warning("txtai LLM not available. Entity extraction will not work.")
            self.llm_available = False
    
    def build(self, documents):
        """
        Build a semantic graph from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Semantic graph
        """
        if not documents:
            return None
        
        # Enable graph in embeddings if not already enabled
        if not hasattr(self.embeddings, "graph") or not self.embeddings.graph:
            # Configure graph in embeddings
            if not hasattr(self.embeddings, "config"):
                self.embeddings.config = {}
            
            self.embeddings.config["graph"] = True
            
            # Apply configuration
            self.embeddings.configure()
        
        # Add documents to embeddings
        self.embeddings.add(documents)
        
        # Index documents
        self.embeddings.index()
        
        # Store reference to graph
        self.graph = self.embeddings.graph
        
        # Return graph
        return self.graph
    
    def save(self, path: str) -> None:
        """
        Save the graph to a file.
        
        Args:
            path: Path to save the graph
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save embeddings (which includes the graph)
        self.embeddings.save(os.path.dirname(path))
        
        self.logger.info(f"Graph saved to {path}")
    
    def load(self, path: str):
        """
        Load a graph from a file.
        
        Args:
            path: Path to load the graph from
            
        Returns:
            Loaded graph
        """
        if not os.path.exists(os.path.dirname(path)):
            raise FileNotFoundError(f"Graph directory not found: {os.path.dirname(path)}")
        
        # Load embeddings (which includes the graph)
        self.embeddings.load(os.path.dirname(path))
        
        # Store reference to graph
        self.graph = self.embeddings.graph
        
        return self.graph


class EntityGraphBuilder(GraphBuilder):
    """
    Build a graph from documents based on entity extraction.
    """
    
    def __init__(self, embeddings, config=None):
        """
        Initialize the entity graph builder.
        
        Args:
            embeddings: Embeddings instance
            config: Configuration dictionary
        """
        self.embeddings = embeddings
        self.config = config or {}
        self.graph = None
        self.logger = logging.getLogger(__name__)
        
        # Check if txtai LLM is available
        try:
            from txtai.pipeline import LLM
            self.llm = LLM("llama2")
            self.llm_available = True
        except ImportError:
            self.logger.warning("txtai LLM not available. Entity extraction will not work.")
            self.llm_available = False
    
    def build(self, documents):
        """
        Build an entity graph from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Entity graph
        """
        if not documents:
            return None
        
        if not self.llm_available:
            self.logger.warning("LLM not available, falling back to semantic graph")
            builder = SemanticGraphBuilder(self.embeddings, self.config)
            return builder.build(documents)
        
        # Enable graph in embeddings if not already enabled
        if not hasattr(self.embeddings, "graph") or not self.embeddings.graph:
            # Configure graph in embeddings
            if not hasattr(self.embeddings, "config"):
                self.embeddings.config = {}
            
            self.embeddings.config["graph"] = True
            self.embeddings.config["graphextractor"] = True
            
            # Apply configuration
            self.embeddings.configure()
        
        # Add documents to embeddings
        self.embeddings.add(documents)
        
        # Index documents
        self.embeddings.index()
        
        # Store reference to graph
        self.graph = self.embeddings.graph
        
        # Return graph
        return self.graph
    
    def save(self, path: str) -> None:
        """
        Save the graph to a file.
        
        Args:
            path: Path to save the graph
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save embeddings (which includes the graph)
        self.embeddings.save(os.path.dirname(path))
        
        self.logger.info(f"Graph saved to {path}")
    
    def load(self, path: str):
        """
        Load a graph from a file.
        
        Args:
            path: Path to load the graph from
            
        Returns:
            Loaded graph
        """
        if not os.path.exists(os.path.dirname(path)):
            raise FileNotFoundError(f"Graph directory not found: {os.path.dirname(path)}")
        
        # Load embeddings (which includes the graph)
        self.embeddings.load(os.path.dirname(path))
        
        # Store reference to graph
        self.graph = self.embeddings.graph
        
        return self.graph


class HybridGraphBuilder(GraphBuilder):
    """
    Build a graph using both semantic similarity and entity extraction.
    """
    
    def __init__(self, embeddings, config=None):
        """
        Initialize the hybrid graph builder.
        
        Args:
            embeddings: Embeddings instance
            config: Configuration dictionary
        """
        self.embeddings = embeddings
        self.config = config or {}
        self.graph = None
        self.logger = logging.getLogger(__name__)
        
        # Create semantic and entity graph builders
        self.semantic_builder = SemanticGraphBuilder(embeddings, config)
        self.entity_builder = EntityGraphBuilder(embeddings, config)
    
    def build(self, documents):
        """
        Build a hybrid graph from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Hybrid graph
        """
        if not documents:
            return None
        
        # Enable graph in embeddings if not already enabled
        if not hasattr(self.embeddings, "graph") or not self.embeddings.graph:
            # Configure graph in embeddings
            if not hasattr(self.embeddings, "config"):
                self.embeddings.config = {}
            
            self.embeddings.config["graph"] = True
            self.embeddings.config["graphextractor"] = True
            
            # Apply configuration
            self.embeddings.configure()
        
        # Add documents to embeddings
        self.embeddings.add(documents)
        
        # Index documents
        self.embeddings.index()
        
        # Store reference to graph
        self.graph = self.embeddings.graph
        
        # Return graph
        return self.graph
    
    def save(self, path: str) -> None:
        """
        Save the graph to a file.
        
        Args:
            path: Path to save the graph
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save embeddings (which includes the graph)
        self.embeddings.save(os.path.dirname(path))
        
        self.logger.info(f"Graph saved to {path}")
    
    def load(self, path: str):
        """
        Load a graph from a file.
        
        Args:
            path: Path to load the graph from
            
        Returns:
            Loaded graph
        """
        if not os.path.exists(os.path.dirname(path)):
            raise FileNotFoundError(f"Graph directory not found: {os.path.dirname(path)}")
        
        # Load embeddings (which includes the graph)
        self.embeddings.load(os.path.dirname(path))
        
        # Store reference to graph
        self.graph = self.embeddings.graph
        
        return self.graph


def create_graph_builder(builder_type: str, embeddings: Embeddings, config: Optional[Dict[str, Any]] = None) -> GraphBuilder:
    """
    Factory function to create a graph builder.

    Args:
        builder_type: Type of graph builder ("semantic", "entity", or "hybrid")
        embeddings: txtai Embeddings instance
        config: Configuration dictionary

    Returns:
        A GraphBuilder instance
    """
    if builder_type == "semantic":
        return SemanticGraphBuilder(embeddings, config)
    elif builder_type == "entity":
        return EntityGraphBuilder(embeddings, config)
    elif builder_type == "hybrid":
        return HybridGraphBuilder(embeddings, config)
    else:
        raise ValueError(f"Unknown graph builder type: {builder_type}")
