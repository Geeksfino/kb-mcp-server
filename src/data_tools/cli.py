#!/usr/bin/env python3
"""
CLI for Knowledge Base operations.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Import txtai
from txtai.app import Application
from txtai.pipeline import Extractor

# Import settings
from data_tools.settings import Settings

# Configure logging
logger = logging.getLogger(__name__)

def setup_logging(debug: bool = False):
    """Set up logging configuration.
    
    Args:
        debug: Whether to enable debug logging
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

def find_config_file() -> Optional[str]:
    """
    Find a configuration file in standard locations.
    
    Returns:
        Path to the configuration file if found, None otherwise.
    """
    # Check environment variable first
    if os.environ.get("KB_CONFIG"):
        config_path = os.environ.get("KB_CONFIG")
        if os.path.exists(config_path):
            return config_path
    
    # Check standard locations
    search_paths = [
        "./config.yaml",
        "./config.yml",
        Path.home() / ".config" / "knowledge-base" / "config.yaml",
        Path.home() / ".config" / "knowledge-base" / "config.yml",
        Path.home() / ".knowledge-base" / "config.yaml",
        Path.home() / ".knowledge-base" / "config.yml",
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return str(path)
    
    return None

def create_application(config_path: Optional[str] = None) -> Application:
    """
    Create a txtai application with the specified configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        txtai.app.Application: Application instance
    """
    # Use provided config if available
    if config_path:
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
        
        if os.path.exists(config_path):
            logger.info(f"Loading configuration from {config_path}")
            try:
                # Create application directly from YAML file path
                app = Application(config_path)
                
                # Log configuration details
                if hasattr(app.embeddings, 'graph') and app.embeddings.graph:
                    logger.info("Graph configuration found in embeddings")
                
                # Log index path
                if hasattr(app, 'config') and 'path' in app.config:
                    logger.info(f"Index will be stored at: {app.config['path']}")
                
                return app
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
                logger.warning("Falling back to default configuration")
        else:
            logger.warning(f"Configuration file not found: {config_path}")
            logger.warning("Falling back to default configuration")
    else:
        logger.info("No configuration file specified, using default configuration")
    
    # If no config provided or loading failed, use default settings
    logger.info("Creating application with default configuration")
    
    # Get settings
    settings = Settings(config_path)
    
    # Create default configuration
    config = {
        "path": ".txtai/index",  # Default index path
        "writable": True,  # Enable index writing
        "content": True,   # Store document content
        "embeddings": {
            "path": settings.get("model_path", "sentence-transformers/all-MiniLM-L6-v2"),
            "gpu": settings.get("model_gpu", True),
            "normalize": settings.get("model_normalize", True),
            "content": True,  # Store document content
            "writable": True   # Enable index writing
        },
        "search": {
            "hybrid": settings.get("hybrid_search", False)
        }
    }
    
    return Application(config)

def build_command(args):
    """
    Handle build command.
    
    Args:
        args: Command-line arguments
    """
    # Use config from args or global args
    config_path = args.config if hasattr(args, 'config') and args.config else args.global_config
    
    if config_path:
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)
            
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            logger.error("Please provide a valid path to a configuration file")
            return
        
        logger.info(f"Using configuration from {config_path}")
    else:
        logger.warning("No configuration file specified, using default settings")
    
    # Create application
    app = create_application(config_path)
    
    # Verify textractor pipeline exists
    if "textractor" not in app.pipelines:
        logger.error("No textractor pipeline configured in YAML. Please add a 'textractor' section to your configuration.")
        logger.error("Example: textractor:\n  paragraphs: true\n  minlength: 100")
        return
    
    # Process documents
    documents = []
    
    # Process JSON input if provided
    if args.json_input:
        try:
            with open(args.json_input, 'r') as f:
                json_data = json.load(f)
                
            # Check if it's a list of documents
            if isinstance(json_data, list):
                documents.extend(json_data)
                logger.info(f"Loaded {len(json_data)} documents from {args.json_input}")
            else:
                logger.error(f"Invalid JSON format in {args.json_input}. Expected a list of documents.")
        except Exception as e:
            logger.error(f"Error loading JSON from {args.json_input}: {e}")
    
    # Process file/directory inputs
    if args.input:
        # Parse extensions
        extensions = None
        if args.extensions:
            # Convert comma-separated string to set of extensions
            extensions = set(ext.strip().lower() for ext in args.extensions.split(","))
            # Add leading dot if not present
            extensions = {ext if ext.startswith('.') else f'.{ext}' for ext in extensions}
        
        for input_path in args.input:
            path = Path(input_path)
            
            if path.is_file():
                logger.info(f"Processing file: {path}")
                try:
                    # Extract text using textractor pipeline
                    segments = app.pipelines["textractor"](str(path))
                    
                    # Create documents with metadata
                    for i, text in enumerate(segments):
                        doc_id = f"{path.stem}_{i}"
                        documents.append({
                            "id": doc_id,
                            "text": text,
                            "metadata": {
                                "source": str(path),
                                "index": i,
                                "total": len(segments)
                            }
                        })
                    logger.info(f"Extracted {len(segments)} segments from {path}")
                except Exception as e:
                    logger.error(f"Error processing file {path}: {e}")
            
            elif path.is_dir():
                logger.info(f"Processing directory: {path}")
                try:
                    # Find all files in directory
                    files = []
                    if extensions:
                        for ext in extensions:
                            files.extend(path.glob(f"**/*{ext}"))
                    else:
                        files = list(path.glob("**/*"))
                    
                    # Filter out directories
                    files = [f for f in files if f.is_file()]
                    
                    logger.info(f"Found {len(files)} files in directory {path}")
                    
                    # Process each file
                    for file_path in files:
                        try:
                            # Extract text using textractor pipeline
                            segments = app.pipelines["textractor"](str(file_path))
                            
                            # Create documents with metadata
                            for i, text in enumerate(segments):
                                doc_id = f"{file_path.stem}_{i}"
                                documents.append({
                                    "id": doc_id,
                                    "text": text,
                                    "metadata": {
                                        "source": str(file_path),
                                        "index": i,
                                        "total": len(segments)
                                    }
                                })
                        except Exception as e:
                            logger.error(f"Error processing file {file_path}: {e}")
                except Exception as e:
                    logger.error(f"Error processing directory {path}: {e}")
            
            else:
                logger.warning(f"Input path not found: {path}")
    
    # Check if we have documents to process
    if not documents:
        logger.error("No documents found to process")
        return
    
    logger.info(f"Processed {len(documents)} documents")
    
    # Use the application's add method which handles both indexing and saving
    logger.info("Indexing documents...")
    try:
        # Add documents to the index
        app.add(documents)
        
        # Build the index
        app.index()
        
        logger.info("Documents indexed successfully")
        
        # Log if graph was built
        if hasattr(app.embeddings, 'graph') and app.embeddings.graph:
            logger.info("Knowledge graph was automatically built based on YAML configuration")
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        return

def retrieve_command(args):
    """
    Handle retrieve command.
    """
    try:
        # Create application
        print(f"Creating application with path: {args.embeddings}")
        app = Application(f"path: {args.embeddings}")

        # Perform search
        print(f"Performing search with query: {args.query}")
        
        # Extract key terms from the query to use for relevance boosting
        query_terms = set(args.query.lower().split())
        # Remove common stop words
        stop_words = {"what", "are", "is", "the", "for", "and", "or", "to", "in", "of", "a", "an"}
        query_terms = query_terms - stop_words
        
        # Perform the search
        results = app.search(args.query, limit=max(10, args.limit * 2), graph=args.graph)  # Get more results initially for filtering

        # Apply generic result enhancement
        if args.graph:
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
                            
                        # Add to candidates
                        nodes_with_scores.append((node_id, score, node["text"]))
                
                # Sort by enhanced score and limit
                nodes_with_scores.sort(key=lambda x: x[1], reverse=True)
                nodes_with_scores = nodes_with_scores[:args.limit]
                
                # Convert to the format expected by format_graph_results
                graph_results = [{"text": text, "score": score} for _, score, text in nodes_with_scores]
            else:
                # Fallback if centrality not available
                graph_results = []
                for x in list(results)[:args.limit]:
                    if "text" in x:
                        graph_results.append({"text": x["text"], "score": x.get("score", 0.5)})
        else:
            # For regular search results, enhance based on query relevance
            enhanced_results = []
            for result in results:
                if "text" in result and "score" in result:
                    # Base score from search
                    score = result["score"]
                    
                    # Boost score based on query term presence
                    text = result["text"].lower()
                    term_matches = sum(1 for term in query_terms if term in text)
                    if term_matches > 0:
                        # Boost proportional to the number of matching terms
                        score *= (1 + (0.1 * term_matches))
                    
                    # Add to candidates
                    enhanced_results.append({"text": result["text"], "score": score})
            
            # Sort by enhanced score and limit
            enhanced_results.sort(key=lambda x: x["score"], reverse=True)
            results = enhanced_results[:args.limit]

        # Print results
        if args.graph:
            # Format and print results
            if graph_results:
                formatted_results = format_graph_results(app.embeddings, graph_results, args.query)
                print(formatted_results)
            else:
                print(f"Q:{args.query}")
                print("No results found.\n")
        else:
            print(f"Results for query: '{args.query}'")
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"  Score: {result['score']:.4f}")
                print(f"  Text: {result['text']}")
                print()

    except Exception as e:
        print(f"Error during retrieval: {e}")
        logger.error(f"Error during retrieval: {e}")

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
        semantic_similarity_threshold = 0.25
        deduplication_threshold = 0.8

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

def generate_command(args):
    """
    Handle generate command with enhanced retrieval capabilities.
    """
    try:
        # Create application
        app = Application(f"path: {args.embeddings}")
        embeddings = app.embeddings

        # Perform enhanced graph search
        results = enhanced_graph_search(embeddings, args.query, limit=args.limit)

        # Format and print results
        if results:
            formatted_results = format_graph_results(embeddings, results, args.query)
            print(formatted_results)
        else:
            print(f"Q:{args.query}")
            print("No results found.\n")

    except Exception as e:
        print(f"Error during generation: {e}")
        logger.error(f"Error during generation: {e}")

def main():
    """
    Main entry point for the Knowledge Base CLI.
    """
    parser = argparse.ArgumentParser(description="Knowledge Base CLI")
    
    # Global arguments
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    subparsers = parser.add_subparsers(title="commands", dest="command", help="Available commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build embeddings database")
    build_parser.add_argument("--input", type=str, nargs="+", help="Path to input files or directories")
    build_parser.add_argument("--extensions", type=str, help="Comma-separated list of file extensions to include")
    build_parser.add_argument("--json_input", type=str, help="Path to JSON file containing a list of documents")
    build_parser.add_argument("--config", type=str, help="Path to configuration file")
    build_parser.set_defaults(func=build_command)
    
    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve information from embeddings database")
    retrieve_parser.add_argument("embeddings", type=str, help="Path to embeddings database")
    retrieve_parser.add_argument("query", type=str, help="Search query")
    retrieve_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    retrieve_parser.add_argument("--graph", action="store_true", help="Enable graph search")
    retrieve_parser.set_defaults(func=retrieve_command)
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate answer using LLM")
    generate_parser.add_argument("embeddings", type=str, help="Path to embeddings database")
    generate_parser.add_argument("query", type=str, help="Search query")
    generate_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
    generate_parser.set_defaults(func=generate_command)
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.debug if hasattr(args, 'debug') else False)
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
