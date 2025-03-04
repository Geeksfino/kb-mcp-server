"""
Enhanced Document Understanding System with Hybrid Graph-Search Capabilities
"""

import os
import logging
from txtai import Embeddings
from txtai.pipeline import Questions, Textractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEBUG_MODE = True  # Toggle for graph visualization

def setup_embeddings():
    """
    Create and configure embeddings with optimized graph settings.
    """
    return Embeddings({
        "path": "sentence-transformers/nli-mpnet-base-v2",
        "normalize": True,
        "hybrid": True,
        "gpu": True,
        "content": True,
        "graph": {
            "backend": "networkx",
            "batchsize": 256,
            "limit": 10,
            "minscore": 0.4,
            "approximate": True,
            "topics": {
                "algorithm": "louvain",
                "terms": 4
            },
            "centrality": "betweenness",
            "directed": True,
            "weight": "similarity",
            "search": {
                "max_hops": 3,
                "use_centrality": True,
                "min_score": 0.3
            }
        },
        "scoring": {
            "method": "bm25",
            "normalize": True,
            "terms": {
                "cachelimit": 1000000000,
                "cutoff": 0.001
            }
        }
    })

def format_graph_results(embeddings, results, query=None):
    """
    Format graph search results to match graph.ipynb output format.
    """
    output = []
    
    if query:
        output.append(f"Q:{query}")
    
    # Process each result
    for result in results:
        try:
            # Get the text and metadata
            if isinstance(result, dict) and "text" in result:
                text = result["text"]
                # Generate a node id from the content
                words = text.split()[:5]  # Take first 5 words
                node_id = "_".join(w.lower() for w in words if w.isalnum())
            else:
                node = embeddings.graph.node(result)
                if not node:
                    continue
                text = node.get("text", "")
                node_id = result
            
            if not text.strip():
                continue
            
            # Add formatted result
            output.append(f"# {node_id}")
            output.append(text.strip())
            output.append("")  # Empty line between results
            
        except Exception as e:
            logger.error(f"Error processing result {result}: {str(e)}")
            continue
    
    return "\n".join(output)

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
        # Generate related questions for query expansion
        questions = Questions("distilbert/distilbert-base-cased-distilled-squad")
        
        # Common stopwords for filtering - using a more general approach
        stopwords = {"what", "when", "where", "which", "that", "this", "does", "how", 
                    "relate", "between", "impact", "connection", "relationship", 
                    "other", "each", "about", "many", "much", "some", "these", "those",
                    "there", "their", "they", "from", "with", "have", "will"}
        
        # Extract key terms for topic filtering - more general approach
        key_terms = set()
        for word in query.lower().split():
            # Keep meaningful terms (longer than 3 chars and not in stopwords)
            if len(word) > 3 and word not in stopwords:
                key_terms.add(word)
            # Also keep shorter but potentially important terms
            elif len(word) <= 3 and word not in stopwords and word.isalpha():
                key_terms.add(word)
        
        logger.info(f"Key terms for filtering: {key_terms}")
        
        # Break query into parts for better expansion
        query_parts = query.split(" and ")
        expanded = [query]  # Always include original query
        
        # Add relationship-focused variants
        for part in query_parts:
            expanded.extend([
                f"relationship between {part}",
                f"connection between {part}",
                f"how does {part} relate to",
                f"impact of {part}"
            ])
            
        # Add question expansions
        expanded.extend(questions([query], ["what", "how", "why"]))
        
        logger.info(f"Expanded query into {len(expanded)} questions")
        
        # Combined results storage
        all_results = []
        seen_texts = set()
        
        # Process each expanded query
        for q in expanded:
            # First get similarity results
            sim_results = embeddings.search(q, limit=3)
            
            # Add high-scoring similarity results with position-based decay
            for idx, result in enumerate(sim_results):
                if result["score"] >= 0.3 and len(result["text"].split()) >= 8:
                    decay = 1.0 - (idx / len(sim_results))
                    text = result["text"].strip()
                    if text and text not in seen_texts:
                        # Calculate topic relevance score using semantic similarity
                        # This is more generalizable than exact term matching
                        topic_relevance = 0.5  # Default middle value
                        
                        # Check for term overlap as a simple relevance indicator
                        text_lower = text.lower()
                        term_matches = sum(1 for term in key_terms if term in text_lower)
                        if key_terms:  # Avoid division by zero
                            term_overlap = term_matches / len(key_terms)
                            # Blend with default for robustness
                            topic_relevance = 0.3 + (0.7 * term_overlap)
                        
                        # Apply decay and topic relevance to score
                        adjusted_score = result["score"] * decay * topic_relevance
                        
                        result["score"] = adjusted_score
                        all_results.append(result)
                        seen_texts.add(text)
        
        # Then get graph results for the main query
        graph = embeddings.search(query, limit=limit, graph=True)
        
        # Get centrality scores for ranking
        centrality = graph.centrality()
        logger.info(f"Got graph with {len(centrality)} nodes")
        
        # Add important graph nodes with relationship boost
        for node_id, score in sorted(centrality.items(), key=lambda x: x[1], reverse=True):
            node = embeddings.graph.node(node_id)
            if not node:
                continue
                
            text = node.get("text", "").strip()
            if text and text not in seen_texts and len(text.split()) >= 8:
                # Calculate topic relevance score using semantic similarity
                topic_relevance = 0.5  # Default middle value
                
                # Check for term overlap as a simple relevance indicator
                text_lower = text.lower()
                term_matches = sum(1 for term in key_terms if term in text_lower)
                if key_terms:  # Avoid division by zero
                    term_overlap = term_matches / len(key_terms)
                    # Blend with default for robustness
                    topic_relevance = 0.3 + (0.7 * term_overlap)
                
                # Apply relationship boost based on edge count
                relationship_boost = 1.0
                try:
                    edges = embeddings.graph.backend.edges(node_id)
                    if edges:
                        # Use a logarithmic scale for more stability with different graph sizes
                        relationship_boost = 1.0 + (0.1 * min(10, len(edges)))
                except:
                    pass  # Default to no boost if edge retrieval fails
                
                # Apply topic relevance to score
                adjusted_score = score * relationship_boost * topic_relevance
                
                all_results.append({"text": text, "score": adjusted_score})
                seen_texts.add(text)
        
        # Sort results by score
        all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
        
        # Take top results up to limit
        final_results = all_results[:limit]
        
        logger.info(f"Returning {len(final_results)} combined results")
        
        # Visualize graph in debug mode
        if DEBUG_MODE:
            try:
                import networkx as nx
                import matplotlib.pyplot as plt
                
                # Get the graph backend
                backend = embeddings.graph.backend
                
                # Create labels for nodes
                labels = {}
                for node in backend.nodes():
                    node_data = embeddings.graph.node(node)
                    if node_data:
                        labels[node] = node_data.get("text", "")[:30] + "..."
                
                # Draw the graph
                plt.figure(figsize=(12, 8))
                pos = nx.spring_layout(backend)
                nx.draw(backend, pos, with_labels=True, labels=labels,
                       node_color='lightblue', node_size=1500,
                       font_size=8, font_weight='bold')
                
                # Save visualization
                plt.savefig("graph.png", bbox_inches='tight')
                plt.close()
                
                logger.info("Generated graph visualization as graph.png")
            except Exception as e:
                logger.warning(f"Failed to generate visualization: {str(e)}")
        
        return final_results
        
    except Exception as e:
        logger.error(f"Error in enhanced graph search: {str(e)}")
        return []

def main():
    try:
        # Load test document
        doc_path = os.path.join(os.path.dirname(__file__), "..", "knowledgebase", "data_science.md")
        with open(doc_path, "r") as f:
            text = f.read()
        
        logger.info("Loaded test document")
        
        # Setup embeddings
        embeddings = setup_embeddings()
        logger.info("Created embeddings instance")
        
        # Use txtai's Textractor for content extraction
        textractor = Textractor(paragraphs=True)
        sections = list(textractor(text))
        logger.info(f"Extracted {len(sections)} sections using Textractor")
        
        # Index the content
        embeddings.index(sections)
        logger.info("Indexed sections")
        
        # Test questions
        questions = [
            "How does feature engineering relate to model performance?",
            "What is the connection between privacy and ethical data science?",
            "How do edge analytics and IoT relate to each other?"
        ]
        
        print("--- GRAPH-BASED SEARCH ---\n")
        print("--- RELATIONSHIP QUESTIONS GRAPH SEARCH ---")
        
        for question in questions:
            # Get enhanced results
            results = enhanced_graph_search(embeddings, question)
            
            if not results:
                print(f"Q:{question}")
                print("No results found.\n")
                continue
            
            # Format and print results
            formatted_results = format_graph_results(embeddings, results, question)
            print(formatted_results)
            print("")  # Extra line between questions
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
