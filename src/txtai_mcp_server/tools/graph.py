"""
Graph-related tools for the txtai MCP server.
"""
import logging
from typing import Dict, List, Optional, Union

from mcp.server.fastmcp import FastMCP, Context
from txtai.graph import GraphFactory

logger = logging.getLogger(__name__)

def register_graph_tools(mcp: FastMCP) -> None:
    """Register graph-related tools."""
    logger.debug("Registering graph tools...")
    
    @mcp.tool(
        name="create_graph",
        description="""Create a semantic knowledge graph to represent relationships between concepts.
        Best used for:
        - Modeling relationships between entities
        - Creating networks of connected information
        - Visualizing concept relationships
        - Building knowledge bases
        
        Example: Create a technology graph:
        nodes=[
            {"id": "py", "text": "Python", "type": "language"},
            {"id": "ds", "text": "Data Science", "type": "field"}
        ]
        relationships=[
            {"source": "py", "target": "ds", "relationship": "used_in"}
        ]"""
    )
    async def create_graph(
        ctx: Context,
        nodes: List[Dict],
        relationships: Optional[List[Dict]] = None,
        backend: str = "networkx"
    ) -> Dict[str, Union[List[Dict], str]]:
        """Implementation of graph creation using txtai graph.
        
        Args:
            nodes: List of node dicts with id and text
            relationships: Optional list of relationship dicts
            backend: Graph storage backend
            
        Returns:
            Dict with nodes, relationships and backend info
        """
        logger.debug(f"Creating graph with {len(nodes)} nodes")
        
        if not ctx.lifespan_context or "txtai_context" not in ctx.lifespan_context:
            raise RuntimeError("TxtAI context not initialized")
            
        txtai_context = ctx.lifespan_context["txtai_context"]
        
        try:
            # Initialize graph if needed
            if not hasattr(txtai_context, "graph"):
                logger.debug("Initializing graph...")
                txtai_context.graph = GraphFactory.create({
                    "backend": backend,
                    "approximate": False
                })
                txtai_context.graph.initialize()
            
            # Add nodes
            for node in nodes:
                txtai_context.graph.addnode(
                    node["id"],
                    text=node["text"],
                    **{k:v for k,v in node.items() if k not in ["id", "text"]}
                )
            
            # Add relationships if provided
            if relationships:
                for rel in relationships:
                    txtai_context.graph.addedge(
                        rel["source"],
                        rel["target"],
                        relationship=rel["relationship"],
                        weight=rel.get("weight", 1.0)
                    )
            
            return {
                "nodes": nodes,
                "relationships": relationships or [],
                "backend": backend
            }
            
        except Exception as e:
            logger.error(f"Graph creation error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
    @mcp.tool(
        name="analyze_graph",
        description="""Analyze the knowledge graph to extract insights about relationships.
        Best used for:
        - Finding important/central nodes
        - Discovering paths between concepts
        - Identifying clusters/topics
        - Understanding network structure
        
        Example: Find path between concepts:
        analysis="path"
        source_id="python"
        target_id="machine_learning" """
    )
    async def analyze_graph(
        ctx: Context,
        analysis: str = "centrality",
        source_id: Optional[str] = None,
        target_id: Optional[str] = None
    ) -> Dict[str, Union[Dict, List]]:
        """Implementation of graph analysis using txtai graph.
        
        Args:
            analysis: Analysis type (centrality, path, topics)
            source_id: Source node for path analysis
            target_id: Target node for path analysis
            
        Returns:
            Dict with analysis results
        """
        logger.debug(f"Analyzing graph: {analysis}")
        
        if not ctx.lifespan_context or "txtai_context" not in ctx.lifespan_context:
            raise RuntimeError("TxtAI context not initialized")
            
        txtai_context = ctx.lifespan_context["txtai_context"]
        
        if not hasattr(txtai_context, "graph"):
            raise RuntimeError("Graph not initialized")
            
        try:
            if analysis == "centrality":
                result = txtai_context.graph.centrality()
                return {"centrality_scores": result}
                
            elif analysis == "path":
                if not source_id or not target_id:
                    raise ValueError("Source and target IDs required for path analysis")
                path = txtai_context.graph.showpath(source_id, target_id)
                return {"path": path}
                
            elif analysis == "topics":
                topics = txtai_context.graph.topics()
                return {"topics": topics}
                
            else:
                raise ValueError(f"Unknown analysis type: {analysis}")
                
        except Exception as e:
            logger.error(f"Graph analysis error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
    @mcp.tool(
        name="extract_graph",
        description="""Automatically extract a knowledge graph from text using AI.
        Best used for:
        - Converting text to structured graphs
        - Discovering entities and relationships
        - Building graphs from documents
        - Understanding document structure
        
        Example: Extract from text:
        text="Python is widely used in data science and machine learning."
        model="gpt-3.5-turbo" (optional) """
    )
    async def extract_graph(
        ctx: Context,
        text: str,
        model: Optional[str] = None
    ) -> Dict[str, Union[List[Dict], str]]:
        """Implementation of graph extraction from text.
        
        Args:
            text: Text to extract from
            model: Optional LLM model to use
            
        Returns:
            Dict with extracted nodes and relationships
        """
        logger.debug("Extracting graph from text")
        
        if not ctx.lifespan_context or "txtai_context" not in ctx.lifespan_context:
            raise RuntimeError("TxtAI context not initialized")
            
        txtai_context = ctx.lifespan_context["txtai_context"]
        
        try:
            # Initialize LLM if needed and model specified
            if model and not hasattr(txtai_context, "llm"):
                from txtai import LLM
                logger.debug(f"Initializing LLM with model: {model}")
                txtai_context.llm = LLM(model)
            
            # Extract entities and relationships
            if hasattr(txtai_context, "llm"):
                # Use LLM for extraction
                prompt = f"""
                Extract an entity relationship graph from the following text. Output as JSON.
                Nodes must have label and type attributes.
                Edges must have source, target and relationship attributes.
                
                text: {text}
                """
                result = txtai_context.llm(prompt)
                
                # Create graph from extracted data
                return await create_graph(ctx, result["nodes"], result["relationships"])
            else:
                # Use basic NER for extraction if no LLM
                if not hasattr(txtai_context, "extractor"):
                    from txtai.pipeline import NER
                    txtai_context.ner = NER()
                
                entities = txtai_context.ner(text)
                nodes = [{"id": i, "text": e[0], "type": e[1]} for i, e in enumerate(entities)]
                
                return await create_graph(ctx, nodes)
                
        except Exception as e:
            logger.error(f"Graph extraction error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
