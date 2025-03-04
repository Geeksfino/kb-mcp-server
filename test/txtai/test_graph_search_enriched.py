"""
Enhanced Document Understanding System with Hybrid Graph-Search Capabilities
"""

import os
import logging
from txtai import Embeddings
from txtai.pipeline import Questions, Textractor, Summary
import re
import time
from functools import lru_cache
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEBUG_MODE = True  # Toggle for graph visualization

# Simple cache implementation with time-based expiration
class SimpleCache:
    """
    A simple cache implementation with time-based expiration.
    """
    def __init__(self, max_size=1000, ttl=3600):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of items to store in the cache
            ttl: Time-to-live in seconds for cache entries
        """
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.access_times = {}
    
    def get(self, key):
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if key in self.cache:
            # Check if the entry has expired
            if time.time() - self.access_times[key] > self.ttl:
                # Remove expired entry
                del self.cache[key]
                del self.access_times[key]
                return None
            
            # Update access time
            self.access_times[key] = time.time()
            return self.cache[key]
        
        return None
    
    def set(self, key, value):
        """
        Add an item to the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # If cache is full, remove the least recently used item
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        # Add new item
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_times.clear()

# Create global cache instances
search_cache = SimpleCache()
similarity_cache = SimpleCache()
summary_cache = SimpleCache()

# Helper function to generate cache keys
def generate_cache_key(*args):
    """
    Generate a cache key from the arguments.
    
    Args:
        *args: Arguments to include in the key
        
    Returns:
        A string key
    """
    # Convert all arguments to strings and join with a separator
    key_parts = [str(arg) for arg in args]
    key_string = "|".join(key_parts)
    
    # Use MD5 to create a fixed-length key
    return hashlib.md5(key_string.encode()).hexdigest()

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

def enhanced_graph_search(embeddings, query, limit=5, enable_context_enrichment=False):
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
        enable_context_enrichment: whether to enrich results with additional context (slower)
        
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
        context_limit = 2  # Reduced number of related sections to find for context enrichment
        context_similarity_threshold = 0.2  # Threshold for finding related context
        summary_max_length = 75  # Reduced maximum length for context summaries
        summary_model = "sshleifer/distilbart-cnn-12-6"  # Use previously downloaded model
        
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
            # Generate cache key
            cache_key = generate_cache_key("similarity", query_text, text)
            
            # Check cache
            cached_result = similarity_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            try:
                # Use txtai's built-in similarity function
                similarity_results = embeddings.similarity(query_text, [text])
                if similarity_results and similarity_results[0]:
                    result = similarity_results[0][1]  # Return the similarity score
                    # Cache the result
                    similarity_cache.set(cache_key, result)
                    return result
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

        # Helper function to enrich results with related context
        def enrich_with_context(results):
            """
            Enhance search results with related context from the knowledge base.
            """
            if not results:
                return results

            # If context enrichment is disabled, return original results
            if not enable_context_enrichment:
                return results
            
            # Import Summary pipeline for generating concise summaries
            summary_pipeline = None
            try:
                # Lazy loading of the summary pipeline with a lighter model
                from txtai.pipeline import Summary
                summary_pipeline = Summary(path=summary_model)
                logger.info(f"Initialized Summary pipeline with model: {summary_model}")
            except Exception as e:
                logger.warning(f"Could not initialize Summary pipeline: {e}")
            
            enriched_results = []
            
            for result in results:
                # Original text and score
                text = result["text"]
                score = result["score"]
                
                # Find related content for context enrichment
                related_sections = []
                try:
                    # Extract key phrases from the result text to use as search queries
                    lines = text.split('\n')
                    search_phrases = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
                    
                    # Limit to first 2-3 non-header lines to avoid too many queries
                    search_phrases = search_phrases[:3]
                    
                    # Search for related content using these phrases
                    for phrase in search_phrases:
                        # Skip very short phrases
                        if len(phrase.split()) < 3:
                            continue
                        
                        related = embeddings.search(phrase, limit=context_limit)
                        for item in related:
                            if (item["score"] >= context_similarity_threshold and 
                                item["text"] != text and 
                                item["text"] not in [r["text"] for r in related_sections]):
                                related_sections.append(item)
                except Exception as e:
                    logger.warning(f"Error finding related content: {e}")
                
                # Sort related sections by score and limit
                related_sections = sorted(related_sections, key=lambda x: x["score"], reverse=True)[:context_limit]
                
                # Generate context summary if we have related sections and summary pipeline
                context_summary = ""
                if related_sections and summary_pipeline:
                    try:
                        # Combine related sections for summarization
                        combined_text = "\n\n".join([r["text"] for r in related_sections])
                        
                        # Generate cache key for summary
                        summary_cache_key = generate_cache_key("summary", combined_text, summary_max_length)
                        
                        # Check cache for summary
                        cached_summary = summary_cache.get(summary_cache_key)
                        if cached_summary is not None:
                            context_summary = cached_summary
                        else:
                            # Generate a concise summary
                            context_summary = summary_pipeline(combined_text, maxlength=summary_max_length)
                            # Cache the summary
                            summary_cache.set(summary_cache_key, context_summary)
                        
                        # Format the summary
                        if context_summary and context_summary.strip():
                            context_summary = f"\n\n**Related Context:**\n{context_summary}"
                    except Exception as e:
                        logger.warning(f"Error generating context summary: {e}")
                
                # Add examples and specific details if available in related sections
                examples = []
                for section in related_sections:
                    section_text = section["text"]
                    # Look for examples, lists, or specific details
                    example_lines = []
                    for line in section_text.split('\n'):
                        line = line.strip()
                        # Identify list items or examples
                        if line.startswith('- ') or line.startswith('* ') or 'example' in line.lower() or 'e.g.' in line.lower():
                            example_lines.append(line)
                    
                    if example_lines:
                        examples.extend(example_lines[:3])  # Limit to 3 examples per section
                
                # Format examples
                examples_text = ""
                if examples:
                    examples_text = "\n\n**Examples:**\n" + "\n".join(examples[:5])  # Limit to 5 examples total
                
                # Combine original text with enriched context
                enriched_text = text
                if context_summary or examples_text:
                    enriched_text = f"{text}{context_summary}{examples_text}"
                
                # Add enriched result
                enriched_results.append({"text": enriched_text, "score": score})
            
            return enriched_results

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
            # Generate cache key for search
            search_cache_key = generate_cache_key("search", q, similarity_threshold, min_word_count)
            
            # Check cache for search results
            cached_results = search_cache.get(search_cache_key)
            if cached_results is not None:
                # Process cached results
                for result in cached_results:
                    text = result["text"].strip()
                    if not text or text in seen_texts or not is_meaningful(text):
                        continue
                    all_results.append(result)
                    seen_texts.add(text)
            else:
                # Perform search
                sim_results = embeddings.search(q, limit=3)
                
                # Process and filter results
                filtered_results = []
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
                        filtered_results.append(result)
                        all_results.append(result)
                        seen_texts.add(text)
                
                # Cache the filtered results
                search_cache.set(search_cache_key, filtered_results)
        
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
        
        # Optionally enrich results with context before finalizing
        final_results = enrich_with_context(all_results[:limit])

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
        # Clear caches before running tests
        search_cache.clear()
        similarity_cache.clear()
        summary_cache.clear()
        
        # Load test document
        doc_path = os.path.join(os.path.dirname(__file__), "..", "knowledgebase", "data_science.md")
        with open(doc_path, "r") as f:
            document = f.read()
        
        logger.info("Loaded test document")
        
        # Create embeddings model with graph
        embeddings = setup_embeddings()
        
        # Extract sections using Textractor
        sections = Textractor(sentences=False, paragraphs=True, join=False, minlength=10)(document)
        
        # Index sections
        embeddings.index([(uid, text, None) for uid, text in enumerate(sections)])
        logger.info(f"Extracted {len(sections)} sections using Textractor")
        logger.info("Indexed sections")
        
        print("--- GRAPH-BASED SEARCH ---")
        
        # Run tests with context enrichment disabled
        print("\n--- BASIC SEARCH (NO CONTEXT ENRICHMENT) ---")
        start_time = time.time()
        
        # Test feature engineering and model performance
        results = enhanced_graph_search(embeddings, "How does feature engineering relate to model performance?", enable_context_enrichment=False)
        print(f"Q:How does feature engineering relate to model performance?")
        print(format_graph_results(embeddings, results))
        
        # Test privacy and ethics
        results = enhanced_graph_search(embeddings, "What is the connection between privacy and ethical data science?", enable_context_enrichment=False)
        print(f"Q:What is the connection between privacy and ethical data science?")
        print(format_graph_results(embeddings, results))
        
        # Test edge analytics and IoT
        results = enhanced_graph_search(embeddings, "How do edge analytics and IoT relate to each other?", enable_context_enrichment=False)
        print(f"Q:How do edge analytics and IoT relate to each other?")
        print(format_graph_results(embeddings, results))
        
        basic_time = time.time() - start_time
        print(f"\nBasic search completed in {basic_time:.2f} seconds")
        
        print("\n--- RELATIONSHIP QUESTIONS GRAPH SEARCH ---")
        start_time = time.time()
        
        # Test feature engineering and model performance
        results = enhanced_graph_search(embeddings, "How does feature engineering relate to model performance?", enable_context_enrichment=True)
        print(f"Q:How does feature engineering relate to model performance?")
        print(format_graph_results(embeddings, results))
        
        # Test privacy and ethics
        results = enhanced_graph_search(embeddings, "What is the connection between privacy and ethical data science?", enable_context_enrichment=True)
        print(f"Q:What is the connection between privacy and ethical data science?")
        print(format_graph_results(embeddings, results))
        
        # Test edge analytics and IoT
        results = enhanced_graph_search(embeddings, "How do edge analytics and IoT relate to each other?", enable_context_enrichment=True)
        print(f"Q:How do edge analytics and IoT relate to each other?")
        print(format_graph_results(embeddings, results))
        
        enriched_time = time.time() - start_time
        print(f"\nEnriched search completed in {enriched_time:.2f} seconds")
        print(f"Context enrichment adds {enriched_time - basic_time:.2f} seconds of processing time")
     
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
