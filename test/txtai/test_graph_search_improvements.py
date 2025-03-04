"""
Enhanced Document Understanding System with Hybrid Graph-Search Capabilities
"""

import os
import logging
from txtai import Embeddings
from txtai.pipeline import Questions, Textractor
import re

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

        if DEBUG_MODE:
            try:
                import networkx as nx
                import matplotlib.pyplot as plt
                backend = embeddings.graph.backend
                labels = {}
                for node in backend.nodes():
                    node_data = embeddings.graph.node(node)
                    if node_data:
                        labels[node] = node_data.get("text", "")[:30] + "..."
                plt.figure(figsize=(12, 8))
                pos = nx.spring_layout(backend)
                nx.draw(backend, pos, with_labels=True, labels=labels,
                        node_color='lightblue', node_size=1500,
                        font_size=8, font_weight='bold')
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
