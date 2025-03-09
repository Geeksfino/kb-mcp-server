"""
Retrieve tools for the txtai MCP server.
"""
import logging
import traceback
import json
from typing import Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import Field
from fast_langdetect import detect as detect_language

from ..core.state import get_txtai_app, get_causal_config
from .causal_config import CausalBoostConfig, DEFAULT_CAUSAL_CONFIG

logger = logging.getLogger(__name__)

def register_retrieve_tools(mcp: FastMCP) -> None:
    """Register retrieve-related tools with the MCP server."""
    logger.debug("Starting registration of retrieve tools...")
    
    @mcp.tool(
        name="retrieve_context",
        description="""
        Retrieve rich contextual information using enhanced graph-based search.
        Best used for:
        - Finding relationships between concepts
        - Building comprehensive context for complex questions
        - Discovering connections in knowledge graphs
        - Understanding how different topics relate to each other
        
        Uses advanced query expansion, semantic similarity, and graph traversal to find the most relevant context.
        
        Example: "How does feature engineering relate to model performance?" will find content explaining the relationship between these concepts.
        """
    )
    async def retrieve_context(
        ctx: Context,
        query: str,
        limit: Optional[int] = Field(5, description="Maximum number of results to return"),
        min_similarity: Optional[float] = Field(0.3, description="Minimum similarity threshold for results")
    ) -> str:
        """
        Retrieve rich contextual information using enhanced graph-based search.
        """
        logger.info(f"Retrieve context request - query: {query}, limit: {limit}, min_similarity: {min_similarity}")
        try:
            # Get the txtai application
            app = get_txtai_app()
            
            # Get causal boost configuration from global state
            causal_config = get_causal_config()
            
            # Log causal boost status
            logger.info("=== Causal Boost Status ===")
            if causal_config:
                logger.info("Causal boost is ENABLED (server configuration)")
                # Log configuration details
                logger.info("Boost multipliers:")
                for boost_type, value in causal_config.boosts.items():
                    logger.info(f"  - {boost_type}: {value}x")
                
                # Detect query language and get patterns
                try:
                    detected_lang_result = detect_language(query)
                    detected_lang = detected_lang_result['lang']
                    logger.info(f"Detected query language: {detected_lang} (confidence: {detected_lang_result['score']:.2f})")
                    patterns = causal_config.get_patterns(detected_lang)
                except Exception as e:
                    logger.warning(f"Language detection failed: {e}. Using default language.")
                    patterns = causal_config.get_patterns()
            else:
                logger.info("Causal boost is DISABLED (server configuration)")
                # Initialize empty patterns when causal boost is disabled
                patterns = {"keywords": set(), "negation": [], "intent": [], "stopwords": set()}
            
            # Extract key terms from the query to use for relevance boosting
            query_terms = set(query.lower().split())
            # Remove stopwords using the configuration
            query_terms = query_terms - patterns.get("stopwords", set())
            
            # Perform the search with graph=True
            # Get more results initially for filtering
            results = app.search(query, limit=max(10, limit * 2), graph=True)
            
            # Check if results is None
            if results is None:
                logger.error(f"Search returned None for query: {query}")
                return json.dumps([{"text": "No results found. The embeddings index may be empty or not properly configured.", "score": 0.0}])
            
            # For graph results, enhance using centrality and query relevance
            if hasattr(results, 'centrality') and callable(results.centrality):
                # Get all nodes with their centrality scores
                nodes_with_scores = []
                for node_id in results.centrality().keys():
                    node = results.node(node_id)
                    if node and "text" in node:
                        # Base score from centrality
                        score = results.centrality()[node_id]
                        
                        # Boost score based on query term presence
                        text = node["text"].lower()
                        term_matches = sum(1 for term in query_terms if term in text)
                        if term_matches > 0:
                            # Boost proportional to the number of matching terms
                            score *= (1 + (0.2 * term_matches))
                            
                            # Apply causal boost if configuration exists and patterns are available
                            if causal_config and patterns:
                                # Check for causal keywords
                                causal_matches = sum(1 for kw in patterns["keywords"] if kw in text)
                                
                                if causal_matches > 0:
                                    # Check for intent phrases
                                    has_causal_intent = any(phrase in query.lower() for phrase in patterns["intent"])
                                    
                                    # Apply appropriate boost
                                    if has_causal_intent:
                                        boost = causal_config.boosts["causal_intent"]
                                        score *= boost
                                        logger.info(f"Applied causal intent boost ({boost}x) to result with score {score}")
                                    else:
                                        boost = causal_config.boosts["general_query"]
                                        score *= boost
                                        logger.info(f"Applied general query boost ({boost}x) to result with score {score}")
                                    
                                    # Additional boost for multiple matches (capped)
                                    if causal_matches > 1:
                                        base_boost = causal_config.boosts["multiple_term"]
                                        actual_boost = min(1.0 + (base_boost * causal_matches),
                                                         1.0 + base_boost * 2)
                                        score *= actual_boost
                                        logger.info(f"Applied multiple term boost ({actual_boost}x) for {causal_matches} matches")
                                    
                                    # Check for negation
                                    if any(neg in text for neg in patterns["negation"]):
                                        boost = causal_config.boosts["negation"]
                                        score *= boost
                                        logger.info(f"Applied negation penalty ({boost}x) to result with score {score}")
                        
                        # Add to candidates if score meets minimum threshold
                        if score >= min_similarity:
                            nodes_with_scores.append((node_id, score, node["text"]))
                
                # Sort by enhanced score and limit
                nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
                nodes_with_scores = nodes_with_scores[:limit]
                
                # Convert to the format expected by format_graph_results
                graph_results = [{"text": text, "score": score} for _, score, text in nodes_with_scores]
            else:
                # Fallback if centrality not available
                graph_results = []
                try:
                    # Try to iterate through results
                    for x in list(results)[:limit]:
                        if "text" in x:
                            graph_results.append({"text": x["text"], "score": x.get("score", 0.5)})
                except (TypeError, AttributeError) as e:
                    # Handle case where results is not iterable
                    logger.error(f"Error iterating through results: {e}")
                    return json.dumps([{"text": "Error processing search results. The search index may not be properly configured.", "score": 0.0}])
            
            # Format results
            if graph_results:
                # Format results for JSON output
                formatted_results = []
                for result in graph_results:
                    formatted_results.append({
                        "text": result["text"],
                        "score": float(result["score"])  # Ensure score is a float for JSON serialization
                    })
                return json.dumps(formatted_results)
            else:
                # Return empty results
                return json.dumps([])
                
        except Exception as e:
            logger.error(f"Error in retrieve context: {str(e)}\n{traceback.format_exc()}")
            # Return a more informative error message in JSON format
            error_message = str(e)
            return json.dumps([{
                "text": f"An error occurred while retrieving context: {error_message}. " +
                        "Please check that the embeddings index is properly configured and not empty.",
                "score": 0.0
            }])

