#!/usr/bin/env python3
"""
Search module for querying the knowledge base.

This module provides a KnowledgeSearch class that enables different types of search
against the knowledge base, including similarity search, exact match, hybrid search,
and graph-based search.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Union, Any, Tuple

from txtai.app import Application

logger = logging.getLogger(__name__)

class KnowledgeSearch:
    """
    Search interface for the knowledge base.
    
    This class provides methods for different types of search against the knowledge base:
    - Similarity search: Using dense embeddings
    - Exact match: Using BM25 sparse indexing
    - Hybrid search: Combining dense and sparse results
    - Graph search: Using the knowledge graph for traversal
    """
    
    def __init__(self, app: Application, graph_path: Optional[str] = None):
        """
        Initialize the KnowledgeSearch with a txtai Application.
        
        Args:
            app: A configured txtai Application instance
            graph_path: Path to a saved knowledge graph (optional)
        """
        self.app = app
        
        # Check if hybrid search is enabled
        self.hybrid_enabled = self.app.config.get("embeddings", {}).get("hybrid", False)
        
        # Check if graph search is enabled in txtai
        self.graph_enabled = hasattr(self.app, "graph") and self.app.graph is not None
        
        # Load external graph if path provided (for legacy graph search)
        self.graph = None
        if graph_path:
            self._load_graph(graph_path)
    
    def similarity_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search using dense vectors.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of document dictionaries with text, metadata, and score
        """
        logger.info(f"Performing similarity search for query: {query}")
        
        # Use txtai's search with weights parameter set to use only dense vectors
        # [1.0, 0.0] means 100% dense vectors, 0% sparse vectors
        # If hybrid is not enabled, this will automatically use only dense vectors
        weights = [1.0, 0.0] if self.hybrid_enabled else None
        results = self.app.search(query, limit=limit, weights=weights)
        
        # Enhance results with metadata
        self._enhance_results(results)
        
        return results
    
    def exact_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform exact match search using BM25 sparse indexing.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of document dictionaries with text, metadata, and score
        """
        if not self.hybrid_enabled:
            logger.warning("Exact search requires hybrid indexing to be enabled")
            return []
        
        logger.info(f"Performing exact match search for query: {query}")
        
        # Use txtai's search with weights parameter set to use only sparse/BM25
        # [0.0, 1.0] means 0% dense vectors, 100% sparse vectors (BM25)
        results = self.app.search(query, limit=limit, weights=[0.0, 1.0])
        
        # Enhance results with metadata
        self._enhance_results(results)
        
        return results
    
    def hybrid_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (combining dense and sparse vectors).
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of document dictionaries with text, metadata, and score
        """
        if not self.hybrid_enabled:
            logger.warning("Hybrid search requires hybrid indexing to be enabled, falling back to similarity search")
            return self.similarity_search(query, limit)
        
        logger.info(f"Performing hybrid search for query: {query}")
        
        # Use txtai's search with balanced weights for dense and sparse vectors
        # [0.5, 0.5] means 50% dense vectors, 50% sparse vectors
        results = self.app.search(query, limit=limit, weights=[0.5, 0.5])
        
        # Enhance results with metadata
        self._enhance_results(results)
        
        return results
    
    def graph_search(self, query: str, limit: int = 5, depth: int = 2) -> List[Dict[str, Any]]:
        """
        Perform graph-based search using knowledge graph traversal.
        
        This method uses either txtai's built-in graph search (if available) or 
        falls back to our custom graph traversal implementation.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            depth: Maximum depth to traverse in the graph (only used for custom graph search)
            
        Returns:
            List of document dictionaries with text, metadata, and score
        """
        logger.info(f"Performing graph search for query: {query}")
        
        # Check if txtai graph is available
        if self.graph_enabled:
            # Use txtai's built-in graph search
            logger.info("Using txtai's built-in graph search")
            results = self.app.search(query, limit=limit, graph=True)
            self._enhance_results(results)
            return results
        
        # Fall back to custom graph search if we have a graph loaded
        elif self.graph is not None:
            logger.info("Using custom graph traversal")
            # First get initial results using hybrid search
            initial_results = self.hybrid_search(query, limit=min(limit, 3))
            
            # Then expand results using graph traversal
            results = self._expand_results_with_graph(initial_results, depth, limit)
            return results
        
        # No graph available, fall back to hybrid search
        else:
            logger.warning("Graph search requires a knowledge graph, falling back to hybrid search")
            return self.hybrid_search(query, limit)
    
    def extractive_qa(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform extractive question answering.
        
        This method uses txtai's extractive QA capabilities to extract
        specific answers from the documents.
        
        Args:
            query: Question to answer
            limit: Maximum number of results to return
            
        Returns:
            List of answer dictionaries with text, metadata, and score
        """
        logger.info(f"Performing extractive QA for question: {query}")
        
        # Check if extractor is available
        if not hasattr(self.app, "extractor") or not self.app.extractor:
            logger.warning("Extractive QA requires an extractor to be configured")
            return self.hybrid_search(query, limit)
        
        # Use txtai's extractive QA
        results = self.app.extractor([(query, None)], limit=limit)
        
        # Flatten results if necessary
        if results and isinstance(results, list) and len(results) == 1:
            results = results[0]
        
        # Enhance results with metadata
        self._enhance_results(results)
        
        return results
    
    def _enhance_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Enhance search results with additional information.
        
        Args:
            results: List of search results to enhance
        """
        for result in results:
            # Log the raw result for debugging
            logger.debug(f"Raw search result: {result}")
            
            # Parse metadata if it's a JSON string
            if "metadata" in result and isinstance(result["metadata"], str):
                try:
                    result["metadata"] = json.loads(result["metadata"])
                except json.JSONDecodeError:
                    pass
            
            # Check if text field is missing or contains only "metadata"
            if "text" not in result or result.get("text") == "metadata":
                # Try to get the actual text from the result data
                if "data" in result and isinstance(result["data"], dict):
                    # Extract text from data dictionary
                    if "text" in result["data"]:
                        result["text"] = result["data"]["text"]
                        logger.debug(f"Extracted text from data: {result['text'][:100]}...")
                    
                    # Extract metadata from data dictionary if available
                    if "metadata" in result["data"] and "metadata" not in result:
                        result["metadata"] = result["data"]["metadata"]
            
            # Add snippet if not present
            if "text" in result and "snippet" not in result:
                result["snippet"] = result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
    
    def _load_graph(self, graph_path: str) -> None:
        """
        Load the knowledge graph.
        
        Args:
            graph_path: Path to the saved graph
        """
        try:
            from .kg import KnowledgeGraph
            kg = KnowledgeGraph(self.app)
            self.graph = kg.load_graph(graph_path)
            logger.info(f"Loaded knowledge graph with {self.graph.number_of_nodes()} nodes")
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
            self.graph = None
    
    def _expand_results_with_graph(self, initial_results: List[Dict[str, Any]], 
                                  depth: int, limit: int) -> List[Dict[str, Any]]:
        """
        Expand search results using graph traversal.
        
        Args:
            initial_results: Initial search results
            depth: Maximum depth to traverse
            limit: Maximum number of results to return
            
        Returns:
            Expanded list of search results
        """
        import networkx as nx
        
        # Track seen document IDs to avoid duplicates
        seen_ids = set()
        expanded_results = []
        
        # Add initial results
        for result in initial_results:
            if result["id"] not in seen_ids:
                seen_ids.add(result["id"])
                expanded_results.append(result)
        
        # Traverse graph to find related documents
        for result in initial_results:
            if result["id"] not in self.graph:
                continue
                
            # Get neighbors up to specified depth
            neighbors = set()
            for d in range(1, depth + 1):
                # Get d-hop neighbors
                d_neighbors = nx.single_source_shortest_path_length(self.graph, result["id"], cutoff=d)
                neighbors.update(d_neighbors.keys())
            
            # Remove the source node itself
            neighbors.discard(result["id"])
            
            # Get documents for neighbors
            for neighbor_id in neighbors:
                if neighbor_id in seen_ids:
                    continue
                    
                # Get document from index
                try:
                    # Use the document ID to retrieve the document
                    neighbor_docs = self.app.search(f"id:{neighbor_id}", limit=1)
                    if neighbor_docs:
                        seen_ids.add(neighbor_id)
                        expanded_results.append(neighbor_docs[0])
                except Exception as e:
                    logger.error(f"Error retrieving document {neighbor_id}: {e}")
                
                # Stop if we've reached the limit
                if len(expanded_results) >= limit:
                    break
            
            # Stop if we've reached the limit
            if len(expanded_results) >= limit:
                break
        
        return expanded_results[:limit]
