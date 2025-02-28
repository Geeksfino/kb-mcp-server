"""
Visualization module for knowledge graphs.

This module implements visualization tools for knowledge graphs:
1. GraphVisualizer: Provides visualization of knowledge graphs
2. VisualizationOptions: Configuration options for visualization
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Union, Any, Tuple, Set

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class VisualizationOptions:
    """Configuration options for visualization."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the VisualizationOptions.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Node options
        self.node_size = self.config.get("node_size", 750)
        self.node_color = self.config.get("node_color", "#0277bd")
        self.node_alpha = self.config.get("node_alpha", 1.0)
        
        # Edge options
        self.edge_color = self.config.get("edge_color", "#454545")
        self.edge_alpha = self.config.get("edge_alpha", 0.5)
        self.edge_width = self.config.get("edge_width", 1.0)
        
        # Label options
        self.font_size = self.config.get("font_size", 8)
        self.font_color = self.config.get("font_color", "#ffffff")
        
        # Layout options
        self.layout = self.config.get("layout", "spring")
        self.layout_seed = self.config.get("layout_seed", 42)
        
        # Figure options
        self.figsize = self.config.get("figsize", (12, 8))
        self.title = self.config.get("title", "Knowledge Graph")
        self.title_fontsize = self.config.get("title_fontsize", 16)
        
        # Community options
        self.community_colors = self.config.get("community_colors", [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ])


class GraphVisualizer:
    """Provides visualization of knowledge graphs."""

    def __init__(self, graph: nx.Graph, options: Optional[VisualizationOptions] = None):
        """
        Initialize the GraphVisualizer.

        Args:
            graph: NetworkX graph
            options: Visualization options
        """
        self.graph = graph
        self.options = options or VisualizationOptions()
        
        # Check if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
        except ImportError:
            logger.warning("Matplotlib not available. Visualization will not work.")
            self.plt = None

    def visualize_graph(self, output_path: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Visualize the entire graph.

        Args:
            output_path: Path to save the visualization (if None, display inline)
            **kwargs: Additional visualization options

        Returns:
            Path to the saved visualization if output_path is provided, None otherwise
        """
        if self.plt is None:
            raise ImportError("Matplotlib not available. Cannot visualize graph.")
        
        # Create figure
        plt = self.plt
        fig, ax = plt.subplots(figsize=self.options.figsize)
        
        # Get layout
        pos = self._get_layout(**kwargs)
        
        # Get node labels
        labels = self._get_node_labels(**kwargs)
        
        # Draw graph
        nx.draw_networkx(
            self.graph,
            pos=pos,
            ax=ax,
            with_labels=True,
            labels=labels,
            node_size=self.options.node_size,
            node_color=self.options.node_color,
            edge_color=self.options.edge_color,
            alpha=self.options.node_alpha,
            width=self.options.edge_width,
            font_size=self.options.font_size,
            font_color=self.options.font_color
        )
        
        # Set title
        ax.set_title(self.options.title, fontsize=self.options.title_fontsize)
        
        # Remove axis
        ax.set_axis_off()
        
        # Save or display
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            return output_path
        else:
            plt.tight_layout()
            plt.show()
            return None

    def visualize_path(self, path: List[str], output_path: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Visualize a specific path.

        Args:
            path: List of node IDs in the path
            output_path: Path to save the visualization (if None, display inline)
            **kwargs: Additional visualization options

        Returns:
            Path to the saved visualization if output_path is provided, None otherwise
        """
        if self.plt is None:
            raise ImportError("Matplotlib not available. Cannot visualize graph.")
        
        # Create subgraph with path nodes
        path_set = set(path)
        subgraph = self.graph.subgraph(path_set)
        
        # Create figure
        plt = self.plt
        fig, ax = plt.subplots(figsize=self.options.figsize)
        
        # Get layout
        pos = self._get_layout(graph=subgraph, **kwargs)
        
        # Get node labels
        labels = self._get_node_labels(graph=subgraph, **kwargs)
        
        # Get path edges
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            subgraph,
            pos=pos,
            ax=ax,
            node_size=self.options.node_size,
            node_color=self.options.node_color,
            alpha=self.options.node_alpha
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            subgraph,
            pos=pos,
            ax=ax,
            width=self.options.edge_width,
            edge_color=self.options.edge_color,
            alpha=self.options.edge_alpha
        )
        
        # Highlight path edges
        nx.draw_networkx_edges(
            subgraph,
            pos=pos,
            ax=ax,
            edgelist=path_edges,
            width=self.options.edge_width * 2,
            edge_color="red",
            alpha=1.0
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            subgraph,
            pos=pos,
            ax=ax,
            labels=labels,
            font_size=self.options.font_size,
            font_color=self.options.font_color
        )
        
        # Set title
        title = kwargs.get("title", f"Path in Knowledge Graph ({len(path)} nodes)")
        ax.set_title(title, fontsize=self.options.title_fontsize)
        
        # Remove axis
        ax.set_axis_off()
        
        # Save or display
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            return output_path
        else:
            plt.tight_layout()
            plt.show()
            return None

    def visualize_community(self, community: Set[str], output_path: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Visualize a community.

        Args:
            community: Set of node IDs in the community
            output_path: Path to save the visualization (if None, display inline)
            **kwargs: Additional visualization options

        Returns:
            Path to the saved visualization if output_path is provided, None otherwise
        """
        if self.plt is None:
            raise ImportError("Matplotlib not available. Cannot visualize graph.")
        
        # Create subgraph with community nodes
        subgraph = self.graph.subgraph(community)
        
        # Create figure
        plt = self.plt
        fig, ax = plt.subplots(figsize=self.options.figsize)
        
        # Get layout
        pos = self._get_layout(graph=subgraph, **kwargs)
        
        # Get node labels
        labels = self._get_node_labels(graph=subgraph, **kwargs)
        
        # Draw graph
        nx.draw_networkx(
            subgraph,
            pos=pos,
            ax=ax,
            with_labels=True,
            labels=labels,
            node_size=self.options.node_size,
            node_color=self.options.node_color,
            edge_color=self.options.edge_color,
            alpha=self.options.node_alpha,
            width=self.options.edge_width,
            font_size=self.options.font_size,
            font_color=self.options.font_color
        )
        
        # Set title
        title = kwargs.get("title", f"Community in Knowledge Graph ({len(community)} nodes)")
        ax.set_title(title, fontsize=self.options.title_fontsize)
        
        # Remove axis
        ax.set_axis_off()
        
        # Save or display
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            return output_path
        else:
            plt.tight_layout()
            plt.show()
            return None

    def visualize_communities(self, communities: List[Set[str]], output_path: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Visualize communities in the graph.

        Args:
            communities: List of communities (each community is a set of node IDs)
            output_path: Path to save the visualization (if None, display inline)
            **kwargs: Additional visualization options

        Returns:
            Path to the saved visualization if output_path is provided, None otherwise
        """
        if self.plt is None:
            raise ImportError("Matplotlib not available. Cannot visualize graph.")
        
        # Create figure
        plt = self.plt
        fig, ax = plt.subplots(figsize=self.options.figsize)
        
        # Get layout
        pos = self._get_layout(**kwargs)
        
        # Get node labels
        labels = self._get_node_labels(**kwargs)
        
        # Create node color map
        node_colors = []
        for node in self.graph.nodes():
            for i, community in enumerate(communities):
                if node in community:
                    color_idx = i % len(self.options.community_colors)
                    node_colors.append(self.options.community_colors[color_idx])
                    break
            else:
                node_colors.append("#cccccc")  # Default color for nodes not in any community
        
        # Draw graph
        nx.draw_networkx(
            self.graph,
            pos=pos,
            ax=ax,
            with_labels=True,
            labels=labels,
            node_size=self.options.node_size,
            node_color=node_colors,
            edge_color=self.options.edge_color,
            alpha=self.options.node_alpha,
            width=self.options.edge_width,
            font_size=self.options.font_size,
            font_color=self.options.font_color
        )
        
        # Set title
        title = kwargs.get("title", f"Communities in Knowledge Graph ({len(communities)} communities)")
        ax.set_title(title, fontsize=self.options.title_fontsize)
        
        # Remove axis
        ax.set_axis_off()
        
        # Save or display
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            return output_path
        else:
            plt.tight_layout()
            plt.show()
            return None

    def export_to_html(self, output_path: str, **kwargs) -> str:
        """
        Export the visualization to an interactive HTML file.

        Args:
            output_path: Path to save the HTML file
            **kwargs: Additional visualization options

        Returns:
            Path to the saved HTML file
        """
        try:
            import pyvis.network as net
        except ImportError:
            raise ImportError("Pyvis not available. Cannot export to HTML.")
        
        # Create network
        network = net.Network(
            height="750px",
            width="100%",
            notebook=False,
            directed=isinstance(self.graph, nx.DiGraph),
            bgcolor="#ffffff",
            font_color="#000000"
        )
        
        # Get node labels
        labels = self._get_node_labels(**kwargs)
        
        # Add nodes
        for node in self.graph.nodes():
            label = labels.get(node, str(node))
            
            # Get node attributes
            node_data = self.graph.nodes[node]
            title = node_data.get("text", label)
            
            # Add node
            network.add_node(
                node,
                label=label,
                title=title,
                color=self.options.node_color
            )
        
        # Add edges
        for u, v, data in self.graph.edges(data=True):
            # Get edge attributes
            title = data.get("type", "")
            weight = data.get("weight", 1.0)
            
            # Add edge
            network.add_edge(
                u,
                v,
                title=title,
                width=weight * self.options.edge_width,
                color=self.options.edge_color
            )
        
        # Set physics options
        network.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {
                    "enabled": true,
                    "iterations": 1000,
                    "updateInterval": 25
                }
            }
        }
        """)
        
        # Save to HTML
        network.save_graph(output_path)
        
        return output_path

    def _get_layout(self, graph: Optional[nx.Graph] = None, **kwargs) -> Dict[str, Tuple[float, float]]:
        """
        Get layout for the graph.

        Args:
            graph: Graph to get layout for (default: self.graph)
            **kwargs: Additional layout options

        Returns:
            Dictionary mapping node IDs to positions
        """
        graph = graph or self.graph
        layout = kwargs.get("layout", self.options.layout)
        seed = kwargs.get("layout_seed", self.options.layout_seed)
        
        if layout == "spring":
            return nx.spring_layout(graph, seed=seed)
        elif layout == "circular":
            return nx.circular_layout(graph)
        elif layout == "kamada_kawai":
            return nx.kamada_kawai_layout(graph)
        elif layout == "spectral":
            return nx.spectral_layout(graph)
        elif layout == "shell":
            return nx.shell_layout(graph)
        elif layout == "spiral":
            return nx.spiral_layout(graph)
        else:
            return nx.spring_layout(graph, seed=seed)

    def _get_node_labels(self, graph: Optional[nx.Graph] = None, **kwargs) -> Dict[str, str]:
        """
        Get labels for nodes.

        Args:
            graph: Graph to get labels for (default: self.graph)
            **kwargs: Additional label options

        Returns:
            Dictionary mapping node IDs to labels
        """
        graph = graph or self.graph
        label_field = kwargs.get("label_field", "id")
        
        labels = {}
        for node in graph.nodes():
            node_data = graph.nodes[node]
            
            # Get label from node data if available
            if label_field in node_data:
                labels[node] = str(node_data[label_field])
            else:
                # Use node ID as label
                labels[node] = str(node)
            
            # Truncate long labels
            max_length = kwargs.get("max_label_length", 20)
            if len(labels[node]) > max_length:
                labels[node] = labels[node][:max_length] + "..."
        
        return labels
