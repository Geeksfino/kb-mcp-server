#!/usr/bin/env python3
"""
Knowledge graph builder for document relationships.

This module provides a KnowledgeGraph class that builds a graph representation
of document relationships for semantic graph traversal and community detection.
"""

import os
import logging
import json
import pickle
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import networkx as nx
from txtai.app import Application

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Knowledge graph builder for document relationships.
    
    This class builds a graph representation of document relationships
    for semantic graph traversal and community detection.
    """
    
    def __init__(self, app: Application):
        """
        Initialize the KnowledgeGraph with a txtai Application.
        
        Args:
            app: A configured txtai Application instance
        """
        self.app = app
        
        # Extract graph configuration
        self.graph_config = self.app.config.get("graph", {})
        
        # Set similarity threshold for creating edges
        self.similarity_threshold = self.graph_config.get("similarity", 0.75)
        
        # Set limit for similar documents
        self.similarity_limit = self.graph_config.get("limit", 10)
        
        # Initialize graph
        self.graph = nx.Graph()
        
        # Track if graph is built
        self.is_built = False
    
    def build_graph(self) -> nx.Graph:
        """
        Build the knowledge graph from the indexed documents.
        
        This method creates a graph where:
        - Nodes are documents
        - Edges represent semantic similarity above the threshold
        
        Returns:
            NetworkX graph object
        """
        logger.info("Building knowledge graph from indexed documents")
        
        # Get all document IDs
        all_docs = self._get_all_documents()
        
        if not all_docs:
            logger.warning("No documents found to build graph")
            return self.graph
        
        logger.info(f"Found {len(all_docs)} documents to process")
        
        # Build the graph
        for i, doc in enumerate(all_docs):
            # Add the document as a node
            self._add_node(doc)
            
            # Find similar documents
            similar_docs = self._find_similar_documents(doc["id"], doc["text"])
            
            # Add edges to similar documents
            for similar_doc, similarity in similar_docs:
                if similar_doc["id"] != doc["id"]:  # Avoid self-loops
                    self._add_edge(doc, similar_doc, similarity)
            
            # Log progress
            if (i + 1) % 100 == 0 or (i + 1) == len(all_docs):
                logger.info(f"Processed {i + 1}/{len(all_docs)} documents")
        
        # Set flag
        self.is_built = True
        
        # Log graph statistics
        logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def find_communities(self) -> Dict[int, List[str]]:
        """
        Find communities in the knowledge graph.
        
        Returns:
            Dictionary mapping community IDs to lists of document IDs
        """
        if not self.is_built:
            logger.warning("Graph not built yet, building now")
            self.build_graph()
        
        logger.info("Finding communities in knowledge graph")
        
        # Use Louvain method for community detection
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph)
            
            # Group documents by community
            communities = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)
            
            logger.info(f"Found {len(communities)} communities")
            return communities
            
        except ImportError:
            logger.warning("python-louvain package not found, using connected components instead")
            
            # Fall back to connected components
            components = list(nx.connected_components(self.graph))
            communities = {i: list(component) for i, component in enumerate(components)}
            
            logger.info(f"Found {len(communities)} connected components")
            return communities
    
    def find_path(self, source_id: str, target_id: str) -> List[str]:
        """
        Find the shortest path between two documents.
        
        Args:
            source_id: ID of the source document
            target_id: ID of the target document
            
        Returns:
            List of document IDs representing the path
            
        Raises:
            ValueError: If the source or target document is not in the graph
            nx.NetworkXNoPath: If no path exists between the documents
        """
        if not self.is_built:
            logger.warning("Graph not built yet, building now")
            self.build_graph()
        
        # Check if nodes exist
        if source_id not in self.graph:
            raise ValueError(f"Source document {source_id} not found in graph")
        if target_id not in self.graph:
            raise ValueError(f"Target document {target_id} not found in graph")
        
        # Find shortest path
        try:
            path = nx.shortest_path(self.graph, source=source_id, target=target_id, weight='weight')
            logger.info(f"Found path with {len(path)} nodes from {source_id} to {target_id}")
            return path
        except nx.NetworkXNoPath:
            logger.warning(f"No path found from {source_id} to {target_id}")
            raise
    
    def save_graph(self, path: str) -> None:
        """
        Save the knowledge graph to disk.
        
        Args:
            path: Path to save the graph
        """
        if not self.is_built:
            logger.warning("Graph not built yet, building now")
            self.build_graph()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        logger.info(f"Saving knowledge graph to {path}")
        
        # Save as pickle
        with open(path, 'wb') as f:
            pickle.dump(self.graph, f)
    
    def load_graph(self, path: str) -> nx.Graph:
        """
        Load a knowledge graph from disk.
        
        Args:
            path: Path to load the graph from
            
        Returns:
            NetworkX graph object
            
        Raises:
            FileNotFoundError: If the graph file does not exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Graph file not found: {path}")
        
        logger.info(f"Loading knowledge graph from {path}")
        
        # Load from pickle
        with open(path, 'rb') as f:
            self.graph = pickle.load(f)
        
        # Set flag
        self.is_built = True
        
        # Log graph statistics
        logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def _get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents from the index.
        
        Returns:
            List of document dictionaries
        """
        # Use a broad search to get all documents
        # This is a simplification - in a real system, you'd want to batch this
        return self.app.search("", limit=100000)
    
    def _find_similar_documents(self, doc_id: str, text: str) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find documents similar to the given text.
        
        Args:
            doc_id: ID of the document to find similar documents for
            text: Text of the document
            
        Returns:
            List of (document, similarity) tuples
        """
        # Search for similar documents
        results = self.app.search(text, limit=self.similarity_limit)
        
        # Filter by similarity threshold and exclude the document itself
        similar_docs = []
        for result in results:
            if result["id"] != doc_id and result["score"] >= self.similarity_threshold:
                similar_docs.append((result, result["score"]))
        
        return similar_docs
    
    def _add_node(self, document: Dict[str, Any]) -> None:
        """
        Add a document as a node to the graph.
        
        Args:
            document: Document dictionary
        """
        # Extract metadata
        try:
            metadata = json.loads(document.get("metadata", "{}"))
        except json.JSONDecodeError:
            metadata = {}
        
        # Add node with attributes
        self.graph.add_node(
            document["id"],
            text=document.get("text", "")[:100],  # Store preview of text
            metadata=metadata
        )
    
    def _add_edge(self, doc1: Dict[str, Any], doc2: Dict[str, Any], similarity: float) -> None:
        """
        Add an edge between two documents.
        
        Args:
            doc1: First document
            doc2: Second document
            similarity: Similarity score between the documents
        """
        # Add edge with similarity as weight (higher similarity = lower distance)
        self.graph.add_edge(
            doc1["id"],
            doc2["id"],
            weight=1.0 - similarity,  # Convert similarity to distance
            similarity=similarity
        )
