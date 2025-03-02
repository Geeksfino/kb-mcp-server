"""
Graph traversal module for knowledge graph querying and analysis.

This module implements advanced graph traversal capabilities:
1. Graph Traversal: Provides path-based traversal of knowledge graphs
2. Path Query: Parser and executor for Cypher-like path queries
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import networkx as nx
import numpy as np
from txtai.embeddings import Embeddings

logger = logging.getLogger(__name__)


class GraphTraversal:
    """
    Provides path-based traversal of knowledge graphs.
    
    This approach is based on Notebook 58 (Advanced RAG with graph path traversal).
    """

    def __init__(self, embeddings: Embeddings, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GraphTraversal.

        Args:
            embeddings: txtai embeddings instance with graph component
            config: Configuration dictionary
        """
        if not hasattr(embeddings, 'graph') or not embeddings.graph:
            raise ValueError("Embeddings instance must have a graph component")
        
        self.embeddings = embeddings
        self.graph = embeddings.graph
        # Access the NetworkX backend for operations not available in txtai's graph API
        self.graph_backend = self.graph.backend if hasattr(self.graph, 'backend') else None
        self.config = config or {}
        
        # Configuration parameters
        self.max_path_length = self.config.get("max_path_length", 3)
        self.max_paths = self.config.get("max_paths", 10)

    def traverse(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Traverse the graph based on a query.
        
        Args:
            query: Query string
            limit: Maximum number of results
            
        Returns:
            List of results with path and score
        """
        # Use txtai's search to find the starting node
        search_results = self.embeddings.search(query, limit=limit)
        
        if not search_results:
            logger.warning(f"No results found for query: {query}")
            return []
        
        results = []
        for result in search_results:
            # Get the ID of the result
            node_id = result["id"] if "id" in result else str(result.get("id", ""))
            
            # Find paths from this node
            paths = self.find_paths_from_node(node_id)
            
            # Add paths to results
            for path in paths:
                path_text = []
                for node in path:
                    # Get the text for each node in the path
                    node_text = self.graph.attribute(node, "text")
                    if node_text:
                        path_text.append(node_text)
                
                results.append({
                    "path": path,
                    "score": result.get("score", 0.0),
                    "text": " ".join(path_text) if path_text else ""
                })
        
        # Sort by score and limit results
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def query_paths(self, query: str, path_expression: Optional[str] = None) -> List[List[str]]:
        """
        Query the graph using path expressions.

        Args:
            query: Query string
            path_expression: Optional Cypher-like path expression

        Returns:
            List of paths (each path is a list of node IDs)
        """
        if path_expression:
            # Use PathQuery to execute the query
            path_query = PathQuery(path_expression)
            return path_query.execute(self.graph_backend)
        else:
            # Try to infer start and end nodes from query
            start_nodes = self._find_nodes_by_query(query)
            if not start_nodes:
                logger.warning(f"No nodes found for query: {query}")
                return []
            
            # Find paths from start nodes
            paths = []
            for start_node in start_nodes[:3]:  # Limit to top 3 start nodes
                paths.extend(self.find_paths_from_node(start_node, max_hops=self.max_path_length))
            
            return paths[:self.max_paths]
    
    def _find_nodes_by_query(self, query: str) -> List[str]:
        """
        Find nodes that match a query using txtai search.

        Args:
            query: Query string

        Returns:
            List of node IDs
        """
        # Use txtai's search functionality
        results = self.embeddings.search(query, limit=5)
        return [result["id"] if "id" in result else str(result.get("id", "")) for result in results]
    
    def find_paths(self, start_node: str, end_node: str, max_hops: Optional[int] = None) -> List[List[str]]:
        """
        Find paths between start and end nodes.

        Args:
            start_node: Start node ID
            end_node: End node ID
            max_hops: Maximum number of hops (default: self.max_path_length)

        Returns:
            List of paths (each path is a list of node IDs)
        """
        if max_hops is None:
            max_hops = self.max_path_length
        
        # Check if nodes exist by trying to get their neighbors
        try:
            # If the node doesn't exist, this will raise an exception
            self.graph.neighbors(start_node)
            self.graph.neighbors(end_node)
        except:
            return []
        
        # Use the NetworkX backend for path finding
        try:
            paths = list(nx.all_simple_paths(self.graph_backend, start_node, end_node, cutoff=max_hops))
            return paths[:self.max_paths]
        except nx.NetworkXNoPath:
            return []
    
    def find_paths_from_node(self, start_node: str, max_hops: Optional[int] = None) -> List[List[str]]:
        """
        Find paths starting from a node.

        Args:
            start_node: Start node ID
            max_hops: Maximum number of hops (default: self.max_path_length)

        Returns:
            List of paths (each path is a list of node IDs)
        """
        if max_hops is None:
            max_hops = self.max_path_length
        
        # Check if node exists
        try:
            # If the node doesn't exist, this will raise an exception
            self.graph.neighbors(start_node)
        except:
            return []
        
        # BFS to find paths
        paths = []
        visited = {start_node}
        queue = [([start_node], 0)]  # (path, depth)
        
        while queue and len(paths) < self.max_paths:
            path, depth = queue.pop(0)
            current = path[-1]
            
            # Add current path to results
            if depth > 0:  # Skip the start node by itself
                paths.append(path)
            
            # Stop if max depth reached
            if depth >= max_hops:
                continue
            
            # Explore neighbors using txtai's graph API
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((path + [neighbor], depth + 1))
        
        return paths
    
    def centrality_analysis(self) -> Dict[str, float]:
        """
        Identify central nodes in the graph.

        Returns:
            Dictionary mapping node IDs to centrality scores
        """
        # Use eigenvector centrality via NetworkX backend
        try:
            centrality = nx.eigenvector_centrality(self.graph_backend)
            return dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True))
        except nx.PowerIterationFailedConvergence:
            logger.warning("Eigenvector centrality failed to converge. Using degree centrality.")
            centrality = nx.degree_centrality(self.graph_backend)
            return dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True))
    
    def community_detection(self) -> List[Set[str]]:
        """
        Detect communities in the graph.

        Returns:
            List of communities (each community is a set of node IDs)
        """
        # Convert to undirected graph if needed
        if isinstance(self.graph_backend, nx.DiGraph):
            G = self.graph_backend.to_undirected()
        else:
            G = self.graph_backend
        
        # Use Louvain method for community detection
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G)
            
            # Group nodes by community
            communities = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = set()
                communities[community_id].add(node)
            
            return list(communities.values())
        except ImportError:
            logger.warning("python-louvain not installed. Using connected components.")
            return list(nx.connected_components(G))
    
    def get_node_info(self, node_id: str) -> Dict[str, Any]:
        """
        Get information about a node.

        Args:
            node_id: Node ID

        Returns:
            Dictionary with node information
        """
        if not self.graph.exists(node_id):
            return {}
        
        # Get node attributes using txtai's graph API
        node_info = {}
        for attr in ["text", "id", "title"]:
            value = self.graph.attribute(node_id, attr)
            if value:
                node_info[attr] = value
        
        # Add degree information from NetworkX backend
        node_info["degree"] = self.graph_backend.degree(node_id)
        if isinstance(self.graph_backend, nx.DiGraph):
            node_info["in_degree"] = self.graph_backend.in_degree(node_id)
            node_info["out_degree"] = self.graph_backend.out_degree(node_id)
        
        # Add centrality
        centrality = self.centrality_analysis()
        node_info["centrality"] = centrality.get(node_id, 0)
        
        return node_info
    
    def get_edge_info(self, source: str, target: str) -> Dict[str, Any]:
        """
        Get information about an edge.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            Dictionary with edge information
        """
        if not self.graph_backend.has_edge(source, target):
            return {}
        
        # Get edge attributes from NetworkX backend
        edge_info = dict(self.graph_backend[source][target])
        
        return edge_info


class PathQuery:
    """
    Parser and executor for Cypher-like path queries.
    """

    def __init__(self, query_string: str):
        """
        Initialize the PathQuery.

        Args:
            query_string: Cypher-like query string
        """
        self.query_string = query_string
        self.parsed_query = self._parse(query_string)
    
    def _parse(self, query_string: str) -> Dict[str, Any]:
        """
        Parse a Cypher-like query.

        Args:
            query_string: Cypher-like query string

        Returns:
            Parsed query as a dictionary
        """
        # Basic parsing of MATCH P=({id: "X"})-[*1..3]->({id: "Y"}) RETURN P LIMIT 20
        query = {}
        
        # Extract MATCH clause
        match_pattern = r'MATCH\s+P=\(\{([^}]+)\}\)-\[\*(\d+)\.\.(\d+)\]->\(\{([^}]+)\}\)'
        match = re.search(match_pattern, query_string, re.IGNORECASE)
        if match:
            start_props = self._parse_props(match.group(1))
            min_hops = int(match.group(2))
            max_hops = int(match.group(3))
            end_props = self._parse_props(match.group(4))
            
            query["start"] = start_props
            query["end"] = end_props
            query["min_hops"] = min_hops
            query["max_hops"] = max_hops
        
        # Extract LIMIT clause
        limit_pattern = r'LIMIT\s+(\d+)'
        limit_match = re.search(limit_pattern, query_string, re.IGNORECASE)
        if limit_match:
            query["limit"] = int(limit_match.group(1))
        else:
            query["limit"] = 10  # Default limit
        
        return query
    
    def _parse_props(self, props_str: str) -> Dict[str, str]:
        """
        Parse property string.

        Args:
            props_str: Property string (e.g., 'id: "X"')

        Returns:
            Dictionary of properties
        """
        props = {}
        for prop in props_str.split(','):
            key, value = prop.split(':')
            key = key.strip()
            value = value.strip().strip('"\'')
            props[key] = value
        
        return props
    
    def execute(self, graph: nx.Graph) -> List[List[str]]:
        """
        Execute the query on a graph.

        Args:
            graph: NetworkX graph

        Returns:
            List of paths (each path is a list of node IDs)
        """
        # Find start and end nodes
        start_nodes = self._find_nodes(graph, self.parsed_query["start"])
        end_nodes = self._find_nodes(graph, self.parsed_query["end"])
        
        if not start_nodes or not end_nodes:
            return []
        
        # Find paths
        paths = []
        for start_node in start_nodes:
            for end_node in end_nodes:
                try:
                    new_paths = list(nx.all_simple_paths(
                        graph, 
                        start_node, 
                        end_node, 
                        cutoff=self.parsed_query["max_hops"]
                    ))
                    
                    # Filter by min_hops
                    new_paths = [p for p in new_paths if len(p) - 1 >= self.parsed_query["min_hops"]]
                    
                    paths.extend(new_paths)
                except nx.NetworkXNoPath:
                    continue
        
        # Apply limit
        return paths[:self.parsed_query["limit"]]
    
    def _find_nodes(self, graph: nx.Graph, props: Dict[str, str]) -> List[str]:
        """
        Find nodes that match properties.

        Args:
            graph: NetworkX graph
            props: Node properties to match

        Returns:
            List of matching node IDs
        """
        matching_nodes = []
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            
            # Check if all properties match
            match = True
            for key, value in props.items():
                if key not in node_data or str(node_data[key]) != value:
                    match = False
                    break
            
            if match:
                matching_nodes.append(node)
        
        return matching_nodes
    
    def to_string(self) -> str:
        """
        Convert the query back to a string.

        Returns:
            Query string
        """
        return self.query_string
