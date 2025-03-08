"""
Retrieve tools for the txtai MCP server.
"""
import logging
import traceback
import json
from typing import Optional

from mcp.server.fastmcp import FastMCP, Context
from pydantic import Field

from ..core.state import get_txtai_app

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
        min_similarity: Optional[float] = Field(0.3, description="Minimum similarity threshold for results"),
        causal_boost: Optional[bool] = Field(True, description="Boost results with causal relationships")
    ) -> str:
        """
        Retrieve rich contextual information using enhanced graph-based search.
        """
        logger.info(f"Retrieve context request - query: {query}, limit: {limit}, min_similarity: {min_similarity}, causal_boost: {causal_boost}")
        try:
            # Get the txtai application
            app = get_txtai_app()
            
            # Extract key terms from the query to use for relevance boosting
            query_terms = set(query.lower().split())
            # Remove common stop words
            stop_words = {"what", "are", "is", "the", "for", "and", "or", "to", "in", "of", "a", "an"}
            query_terms = query_terms - stop_words
            
            # Detect if the query has causal intent
            has_causal_intent = detect_causal_intent(query) if causal_boost else False
            logger.info(f"Query causal intent detection: {has_causal_intent}")
            
            # Perform the search with graph=True
            # Get more results initially for filtering
            results = app.search(query, limit=max(10, limit * 2), graph=True)
            
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
                            
                            # Apply causal boost if requested and either:
                            # 1. The query has causal intent, or
                            # 2. We're in a fallback mode where causal_boost is enabled but no causal intent was detected
                            if causal_boost:
                                # Expanded causal keywords with more nuanced terms
                                causal_keywords = {
                                    # Original keywords
                                    "causes", "leads to", "improves", "boosts", "results in", "reduces", "enhances",
                                    # Additional keywords
                                    "triggers", "impacts", "drives", "mitigates", "correlates with", "affects",
                                    "contributes to", "influences", "determines", "enables", "facilitates",
                                    "prevents", "inhibits", "accelerates", "slows", "depends on", "relies on"
                                }
                                
                                # Check for causal keywords in the text
                                causal_term_matches = sum(1 for keyword in causal_keywords if keyword in text)
                                
                                # Apply a dynamic boost based on causal intent and keyword matches
                                if causal_term_matches > 0:
                                    # If query has causal intent, apply a stronger boost
                                    if has_causal_intent:
                                        # Apply a stronger boost (1.3x) for queries with explicit causal intent
                                        score *= 1.3
                                    else:
                                        # Apply a milder boost (1.1x) for general queries
                                        score *= 1.1
                                        
                                    # Additional boost for multiple causal terms (up to 1.2x for 3+ terms)
                                    if causal_term_matches > 1:
                                        score *= min(1.0 + (0.1 * causal_term_matches), 1.2)
                                    
                                    # Check for negation patterns that might indicate false causality
                                    negation_patterns = ["not cause", "doesn't cause", "no evidence", "not related to", 
                                                        "doesn't lead", "doesn't result", "doesn't improve"]
                                    if any(pattern in text for pattern in negation_patterns):
                                        # Reduce the boost for texts with negated causality
                                        score *= 0.7
                        
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
                for x in list(results)[:limit]:
                    if "text" in x:
                        graph_results.append({"text": x["text"], "score": x.get("score", 0.5)})
            
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
            return f"Error processing retrieve context: {str(e)}"

    def detect_causal_intent(query):
        """
        Detect if a query has causal intent by looking for causal phrases and question patterns.
        
        Args:
            query: The search query string
            
        Returns:
            bool: True if the query appears to have causal intent, False otherwise
        """
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Causal intent phrases
        causal_phrases = [
            "why", "how does", "what causes", "what leads to", "what results in",
            "reason for", "effect of", "impact of", "influence of", "relationship between",
            "connection between", "correlation between", "cause of", "caused by",
            "leads to", "results in", "affects", "influences", "determines",
            "how can", "how to improve", "how to increase", "how to reduce"
        ]
        
        # Check for causal phrases in the query
        for phrase in causal_phrases:
            if phrase in query_lower:
                return True
        
        # Check for question patterns that often indicate causal relationships
        if query_lower.startswith(("why ", "how ", "what happens when ")):
            return True
            
        # Check for "does X affect Y" pattern
        if "affect" in query_lower or "effect" in query_lower or "impact" in query_lower:
            return True
            
        return False

    def enhanced_graph_search(embeddings, query, limit=5):
        """
        Improved graph search using txtai's built-in graph capabilities with advanced features.
        
        This implementation focuses on:
        1. Advanced query expansion using txtai's Questions pipeline
        2. Improved result fusion with position-based decay and relationship boost
        3. Better topic relevance through semantic similarity rather than exact matching
        4. Proper deduplication and minimum length filtering
        
        Args:
            embeddings: txtai Embeddings instance with graph component
            query: search query string
            limit: maximum number of results to return
            
        Returns:
            List of search results with text and score
        """
        try:
            # Configurable parameters
            similarity_threshold = 0.3
            min_word_count = 8
            min_word_count_fallback = 5
            base_topic_relevance = 0.3
            topic_weight = 0.7
            edge_boost_factor = 0.1
            min_keyterm_matches = 2
            min_centrality = 0.15
            causal_boost = 1.5
            semantic_similarity_threshold = 0.25  # Threshold for semantic similarity filtering
            deduplication_threshold = 0.8  # Threshold for considering two texts as duplicates

            # Define causal keywords
            causal_keywords = {"causes", "leads to", "improves", "boosts", "results in", "reduces", "enhances"}

            # Use more robust stopwords: try nltk, fallback to default
            try:
                from nltk.corpus import stopwords
                stopwords_set = set(stopwords.words('english'))
            except Exception:
                stopwords_set = {"what", "when", "where", "which", "that", "this", "does", "how", 
                                 "relate", "between", "impact", "connection", "relationship", 
                                 "other", "each", "about", "many", "much", "some", "these", "those",
                                 "there", "their", "they", "from", "with", "have", "will"}

            import re
            # Extract key terms using regex
            words = re.findall(r'\w+', query.lower())
            key_terms = {word for word in words if len(word) > 2 and word not in stopwords_set}

            logger.info(f"Key terms for filtering: {key_terms}")

            # Helper function to check if a text is meaningful (not just an empty header)
            def is_meaningful(text):
                stripped = text.strip()
                if not stripped:
                    return False

                # If the text starts with '#', it is likely a header
                if stripped.startswith('#'):
                    # Remove '#' symbols and replace underscores with spaces
                    content = stripped.lstrip('#').strip().replace('_', ' ')

                    # Check if header has at least 2 words
                    if len(content.split()) < 2:
                        return False

                    # Additionally, if the header is only a single line (i.e. no newline), consider it empty
                    if '\n' not in stripped:
                        return False

                return True

            # Helper function to compute semantic similarity between query and text
            def compute_semantic_similarity(text, query_text):
                """
                Compute semantic similarity between text and query using txtai's similarity function.
                Returns a similarity score between 0 and 1.
                """
                try:
                    # Use txtai's built-in similarity function
                    similarity_results = embeddings.similarity(query_text, [text])
                    if similarity_results and similarity_results[0]:
                        return similarity_results[0][1]  # Return the similarity score
                    return 0.0
                except Exception as e:
                    logger.warning(f"Error computing semantic similarity: {e}")
                    return 0.0

            # Helper function to remove duplicate or near-duplicate results
            def remove_duplicates(results, threshold=deduplication_threshold):
                """
                Remove near-duplicate results using semantic similarity.
                Returns a list of unique results.
                """
                if not results:
                    return []
                
                unique_results = []
                unique_texts = []
                
                # Sort by score to keep highest scoring duplicates
                sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
                
                for result in sorted_results:
                    text = result["text"]
                    is_duplicate = False
                    
                    # Check if this text is similar to any existing unique text
                    for unique_text in unique_texts:
                        # Use txtai's built-in similarity to compare texts
                        similarity = compute_semantic_similarity(text, unique_text)
                        if similarity >= threshold:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        unique_results.append(result)
                        unique_texts.append(text)
                
                return unique_results

            # Query expansion: Generate additional formulations using generic relationship variants and Questions pipeline
            from txtai.pipeline import Questions
            questions = Questions("distilbert/distilbert-base-cased-distilled-squad")

            expansion_variants = []
            if key_terms:
                for term in key_terms:
                    expansion_variants.extend([
                        f"How is {term} related?",
                        f"What is the connection of {term}?",
                        f"Explain relationship for {term}"
                    ])

            pipeline_questions = questions([query], ["what", "how", "why"])

            expanded = [query] + expansion_variants + pipeline_questions

            logger.info(f"Expanded query into {len(expanded)} formulations")

            # Combined results storage
            all_results = []
            seen_texts = set()

            # Process each expanded query using similarity search
            for q in expanded:
                sim_results = embeddings.search(q, limit=3)
                for idx, result in enumerate(sim_results):
                    if result["score"] >= similarity_threshold and len(result["text"].split()) >= min_word_count:
                        decay = 1.0 - (idx / len(sim_results))
                        text = result["text"].strip()
                        if not text or text in seen_texts or not is_meaningful(text):
                            continue
                        text_lower = text.lower()
                        term_matches = sum(1 for term in key_terms if term in text_lower)
                        # Only consider result if it has at least min_keyterm_matches
                        if key_terms and term_matches < min_keyterm_matches:
                            continue
                        if key_terms:
                            term_overlap = term_matches / len(key_terms)
                            topic_relevance = base_topic_relevance + (topic_weight * term_overlap)
                        else:
                            topic_relevance = base_topic_relevance

                        adjusted_score = result["score"] * decay * topic_relevance
                        # Apply causal boost if any causal keyword is present
                        if any(causal_kw in text_lower for causal_kw in causal_keywords):
                            adjusted_score *= causal_boost

                        # Apply semantic similarity filtering
                        semantic_similarity = compute_semantic_similarity(text, query)
                        if semantic_similarity < semantic_similarity_threshold:
                            continue

                        # Boost score with semantic similarity
                        adjusted_score *= (1.0 + semantic_similarity)

                        result["score"] = adjusted_score
                        all_results.append(result)
                        seen_texts.add(text)

            # Retrieve graph results for the main query with centrality filtering
            graph = embeddings.search(query, limit=limit, graph=True)
            centrality = graph.centrality()
            logger.info(f"Got graph with {len(centrality)} nodes")

            for node_id, score in sorted(centrality.items(), key=lambda x: x[1], reverse=True):
                if score < min_centrality:
                    continue
                node = embeddings.graph.node(node_id)
                if not node:
                    continue
                text = node.get("text", "").strip()
                if not text or text in seen_texts or not is_meaningful(text):
                    continue
                if len(text.split()) < min_word_count:
                    continue
                text_lower = text.lower()
                term_matches = sum(1 for term in key_terms if term in text_lower)
                if key_terms and term_matches < min_keyterm_matches:
                    continue
                if key_terms:
                    term_overlap = term_matches / len(key_terms)
                    topic_relevance = base_topic_relevance + (topic_weight * term_overlap)
                else:
                    topic_relevance = base_topic_relevance

                relationship_boost = 1.0
                try:
                    edges = embeddings.graph.backend.edges(node_id)
                    if edges:
                        relationship_boost = 1.0 + (edge_boost_factor * min(10, len(edges)))
                except Exception:
                    pass

                adjusted_score = score * relationship_boost * topic_relevance
                # Apply causal boost if causal keywords present in graph node text
                if any(causal_kw in text_lower for causal_kw in causal_keywords):
                    adjusted_score *= causal_boost

                # Apply semantic similarity filtering
                semantic_similarity = compute_semantic_similarity(text, query)
                if semantic_similarity < semantic_similarity_threshold:
                    continue

                # Boost score with semantic similarity
                adjusted_score *= (1.0 + semantic_similarity)

                all_results.append({"text": text, "score": adjusted_score})
                seen_texts.add(text)

            all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
            
            # Apply deduplication before limiting results
            all_results = remove_duplicates(all_results)
            
            final_results = all_results[:limit]

            # Fallback: if we have fewer than limit results, relax min_word_count and search again
            if len(final_results) < limit:
                for q in expanded:
                    sim_results = embeddings.search(q, limit=3)
                    for idx, result in enumerate(sim_results):
                        if result["score"] >= similarity_threshold and len(result["text"].split()) >= min_word_count_fallback:
                            text = result["text"].strip()
                            if not text or text in seen_texts or not is_meaningful(text):
                                continue
                            text_lower = text.lower()
                            term_matches = sum(1 for term in key_terms if term in text_lower)
                            if key_terms and term_matches < min_keyterm_matches:
                                continue
                            if key_terms:
                                term_overlap = term_matches / len(key_terms)
                                topic_relevance = base_topic_relevance + (topic_weight * term_overlap)
                            else:
                                topic_relevance = base_topic_relevance

                            adjusted_score = result["score"] * topic_relevance
                            if any(causal_kw in text_lower for causal_kw in causal_keywords):
                                adjusted_score *= causal_boost

                            # Apply semantic similarity filtering
                            semantic_similarity = compute_semantic_similarity(text, query)
                            if semantic_similarity < semantic_similarity_threshold:
                                continue

                            # Boost score with semantic similarity
                            adjusted_score *= (1.0 + semantic_similarity)

                            result["score"] = adjusted_score
                            all_results.append(result)
                            seen_texts.add(text)
                all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
                
                # Apply deduplication before limiting results
                all_results = remove_duplicates(all_results)
                
                final_results = all_results[:limit]

            return final_results

        except Exception as e:
            logger.error(f"Error in enhanced graph search: {str(e)}")
            return []
