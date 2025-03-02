"""
RAG (Retrieval Augmented Generation) pipeline module.

This module implements a flexible RAG pipeline:
1. RAGPipeline: Coordinates the RAG process
2. Retriever: Defines the interface for context retrieval
3. Generator: Handles LLM-based text generation
4. Citation: Handles citation generation and verification
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

import networkx as nx
import numpy as np
from txtai.embeddings import Embeddings

from .graph_traversal import GraphTraversal

logger = logging.getLogger(__name__)


class Retriever(ABC):
    """Abstract base class for context retrieval."""

    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.

        Args:
            query: Query string
            **kwargs: Additional arguments for retrieval

        Returns:
            List of context documents
        """
        pass


class VectorRetriever(Retriever):
    """Retrieves context based on vector similarity."""

    def __init__(self, embeddings: Embeddings, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the VectorRetriever.

        Args:
            embeddings: txtai Embeddings instance
            config: Configuration dictionary
        """
        self.embeddings = embeddings
        self.config = config or {}
        
        # Configuration parameters
        self.limit = self.config.get("limit", 10)
        self.min_score = self.config.get("min_score", 0.3)

    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query using vector similarity.

        Args:
            query: Query string
            **kwargs: Additional arguments for retrieval
                - limit: Maximum number of results to return
                - min_score: Minimum similarity score

        Returns:
            List of context documents
        """
        # Override config with kwargs if provided
        limit = kwargs.get("limit", self.limit)
        min_score = kwargs.get("min_score", self.min_score)
        
        # Search for relevant documents
        results = self.embeddings.search(query, limit=limit)
        
        # Filter by score
        results = [r for r in results if r["score"] >= min_score]
        
        return results


class GraphRetriever(Retriever):
    """Retrieves context based on graph relationships."""

    def __init__(self, embeddings: Embeddings, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the GraphRetriever.

        Args:
            embeddings: txtai Embeddings instance with graph component
            config: Configuration dictionary
        """
        self.embeddings = embeddings
        
        # Check if embeddings has a graph component
        if not hasattr(embeddings, 'graph') or not embeddings.graph:
            raise ValueError("Embeddings instance must have a graph component")
        
        self.graph = embeddings.graph
        self.config = config or {}
        
        # Configuration parameters
        self.limit = self.config.get("limit", 10)
        self.max_hops = self.config.get("max_hops", 2)
        
        # Create graph traversal
        self.traversal = GraphTraversal(embeddings, {"max_path_length": self.max_hops})

    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query using graph relationships.

        Args:
            query: Query string
            **kwargs: Additional arguments for retrieval
                - limit: Maximum number of results to return
                - max_hops: Maximum number of hops from seed nodes

        Returns:
            List of context documents
        """
        # Override config with kwargs if provided
        limit = kwargs.get("limit", self.limit)
        max_hops = kwargs.get("max_hops", self.max_hops)
        
        # First, find seed nodes using vector similarity
        seed_results = self.embeddings.search(query, limit=5)
        seed_ids = [r["id"] if "id" in r else str(r.get("id", "")) for r in seed_results]
        
        # Then, find nodes connected to seed nodes
        context_ids = set()
        for seed_id in seed_ids:
            try:
                # Check if node exists by trying to get its neighbors
                self.graph.neighbors(seed_id)
                
                # Find paths from seed node
                paths = self.traversal.find_paths_from_node(seed_id, max_hops=max_hops)
                
                # Add all nodes in paths to context
                for path in paths:
                    context_ids.update(path)
            except:
                # Node doesn't exist, skip it
                continue
        
        # Retrieve document data for context nodes
        context = []
        for node_id in context_ids:
            try:
                # Check if node exists by trying to get its attributes
                # Get node attributes using txtai's graph API
                node_data = {}
                for attr in ["text", "id", "title"]:
                    value = self.graph.attribute(node_id, attr)
                    if value:
                        node_data[attr] = value
                
                if "text" in node_data:
                    context.append(node_data)
            except:
                # Node doesn't exist, skip it
                continue
        
        # Sort by relevance to query
        for doc in context:
            if "text" in doc:
                doc["score"] = float(self.embeddings.similarity(query, doc["text"]))
        
        context.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return context[:limit]


class PathRetriever(Retriever):
    """Retrieves context based on path traversal."""

    def __init__(self, embeddings: Embeddings, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the PathRetriever.

        Args:
            embeddings: txtai Embeddings instance with graph component
            config: Configuration dictionary
        """
        self.embeddings = embeddings
        
        # Check if embeddings has a graph component
        if not hasattr(embeddings, 'graph') or not embeddings.graph:
            raise ValueError("Embeddings instance must have a graph component")
        
        self.graph = embeddings.graph
        self.config = config or {}
        
        # Configuration parameters
        self.limit = self.config.get("limit", 10)
        
        # Create graph traversal
        self.traversal = GraphTraversal(embeddings)

    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query using path traversal.

        Args:
            query: Query string
            **kwargs: Additional arguments for retrieval
                - limit: Maximum number of results to return
                - path_expression: Cypher-like path expression

        Returns:
            List of context documents
        """
        # Override config with kwargs if provided
        limit = kwargs.get("limit", self.limit)
        path_expression = kwargs.get("path_expression")
        
        # Find paths
        paths = self.traversal.query_paths(query, path_expression)
        
        # Extract unique nodes from paths
        context_ids = set()
        for path in paths:
            context_ids.update(path)
        
        # Retrieve document data for context nodes
        context = []
        for node_id in context_ids:
            try:
                # Check if node exists by trying to get its attributes
                # Get node attributes using txtai's graph API
                node_data = {}
                for attr in ["text", "id", "title"]:
                    value = self.graph.attribute(node_id, attr)
                    if value:
                        node_data[attr] = value
                
                if "text" in node_data:
                    context.append(node_data)
            except:
                # Node doesn't exist, skip it
                continue
        
        # Sort by relevance to query
        for doc in context:
            if "text" in doc:
                doc["score"] = float(self.embeddings.similarity(query, doc["text"]))
        
        context.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return context[:limit]


class ExactRetriever(Retriever):
    """Retrieves context based on exact text matching."""

    def __init__(self, embeddings: Embeddings, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ExactRetriever.

        Args:
            embeddings: txtai Embeddings instance
            config: Configuration dictionary
        """
        self.embeddings = embeddings
        self.config = config or {}
        
        # Configuration parameters
        self.limit = self.config.get("limit", 10)
        self.case_sensitive = self.config.get("case_sensitive", False)

    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query using exact text matching.

        Args:
            query: Query string
            **kwargs: Additional arguments for retrieval
                - limit: Maximum number of results to return
                - case_sensitive: Whether to perform case-sensitive matching

        Returns:
            List of context documents
        """
        # Override config with kwargs if provided
        limit = kwargs.get("limit", self.limit)
        case_sensitive = kwargs.get("case_sensitive", self.case_sensitive)
        
        # First try vector search to get documents that might contain the query
        vector_retriever = VectorRetriever(self.embeddings)
        results = vector_retriever.retrieve(query, limit=limit * 2)  # Get more results to filter
        
        # Filter results for exact matches
        exact_matches = []
        for result in results:
            if "text" in result and result["text"]:
                # Apply case sensitivity
                doc_text = result["text"] if case_sensitive else result["text"].lower()
                search_query = query if case_sensitive else query.lower()
                
                # Check for exact match
                if search_query in doc_text:
                    # Boost the score for exact matches
                    result["score"] = min(1.0, result["score"] * 1.5)  # Boost score but cap at 1.0
                    exact_matches.append(result)
                    
                    # Stop if we've reached the limit
                    if len(exact_matches) >= limit:
                        break
        
        # If we found exact matches, return them
        if exact_matches:
            return exact_matches
            
        # Otherwise, fall back to vector search results
        logger.info("No exact matches found, falling back to vector search")
        return results[:limit]


class Generator:
    """Handles LLM-based text generation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configuration parameters
        self.model = self.config.get("model", "TheBloke/Mistral-7B-OpenOrca-AWQ")
        self.template = self.config.get("template")
        self.max_tokens = self.config.get("max_tokens", 1024)
        
        # Initialize LLM
        self._init_llm()

    def _init_llm(self):
        """Initialize the LLM for text generation."""
        try:
            from txtai import LLM
            
            # Initialize LLM with template if provided
            if self.template:
                self.llm = LLM(self.model, template=self.template)
            else:
                self.llm = LLM(self.model)
        except ImportError:
            logger.warning("txtai LLM not available. Text generation will not work.")
            self.llm = None

    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate a response using an LLM.

        Args:
            query: Query string
            context: Context documents

        Returns:
            Generated response
        """
        if self.llm is None:
            raise ValueError("LLM not initialized. Cannot generate text.")
        
        # Build context string
        context_text = "\n\n".join(doc.get("text", "") for doc in context)
        
        # Create prompt
        if self.template:
            # Template is handled by the LLM
            prompt = f"""
Answer the following question using only the context below. Only include information specifically discussed.

question: {query}
context: {context_text}
"""
        else:
            # Create a default prompt
            prompt = f"""You are a friendly assistant. You answer questions from users.

Answer the following question using only the context below. Only include information specifically discussed.

question: {query}
context: {context_text}  
"""
        
        # Generate response
        response = self.llm(prompt, maxlength=self.max_tokens)
        
        return response


class Citation:
    """Handles citation generation and verification."""

    def __init__(self, embeddings: Embeddings, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Citation.

        Args:
            embeddings: txtai Embeddings instance
            config: Configuration dictionary
        """
        self.embeddings = embeddings
        self.config = config or {}
        
        # Configuration parameters
        self.method = self.config.get("method", "semantic_similarity")
        self.threshold = self.config.get("threshold", 0.7)

    def find_sources(self, response: str, context: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find sources for statements in the response.

        Args:
            response: Generated response
            context: Context documents

        Returns:
            Dictionary mapping statements to sources
        """
        # Split response into sentences
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        sentences = nltk.sent_tokenize(response)
        
        # Find sources for each sentence
        sources = {}
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip short sentences
                continue
            
            if self.method == "semantic_similarity":
                # Find most similar context documents
                matches = []
                for doc in context:
                    text = doc.get("text", "")
                    if text:
                        similarity = float(self.embeddings.similarity(sentence, text))
                        if similarity >= self.threshold:
                            matches.append({
                                "doc": doc,
                                "score": similarity
                            })
                
                # Sort by similarity
                matches.sort(key=lambda x: x["score"], reverse=True)
                
                if matches:
                    sources[sentence] = [m["doc"] for m in matches[:3]]
            
            elif self.method == "keyword_matching":
                # Simple keyword matching
                matches = []
                sentence_words = set(sentence.lower().split())
                
                for doc in context:
                    text = doc.get("text", "")
                    if text:
                        text_words = set(text.lower().split())
                        overlap = len(sentence_words.intersection(text_words)) / len(sentence_words)
                        
                        if overlap >= self.threshold:
                            matches.append({
                                "doc": doc,
                                "score": overlap
                            })
                
                # Sort by overlap
                matches.sort(key=lambda x: x["score"], reverse=True)
                
                if matches:
                    sources[sentence] = [m["doc"] for m in matches[:3]]
        
        return sources
    
    def verify_response(self, response: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify the response against the context.

        Args:
            response: Generated response
            context: Context documents

        Returns:
            Verification results
        """
        # Find sources
        sources = self.find_sources(response, context)
        
        # Calculate coverage
        total_sentences = len(nltk.sent_tokenize(response))
        sourced_sentences = len(sources)
        
        coverage = sourced_sentences / total_sentences if total_sentences > 0 else 0
        
        return {
            "sources": sources,
            "coverage": coverage,
            "verified": coverage >= 0.8  # Consider verified if 80% of sentences have sources
        }


class RAGPipeline:
    """Coordinates the RAG process."""

    def __init__(self, embeddings: Embeddings, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAGPipeline.

        Args:
            embeddings: txtai Embeddings instance
            config: Configuration dictionary
        """
        self.embeddings = embeddings
        self.config = config or {}
        
        # Get graph if available
        self.graph = getattr(embeddings, "graph", None)
        
        # Create components
        self.retriever = self._create_retriever()
        self.generator = Generator(self.config.get("generator"))
        
        # Create citation handler if enabled
        citation_config = self.config.get("citation", {})
        if citation_config.get("enabled", False):
            self.citation = Citation(embeddings, citation_config)
        else:
            self.citation = None

    def _create_retriever(self) -> Retriever:
        """
        Create the appropriate retriever based on configuration.

        Returns:
            A Retriever instance
        """
        retriever_type = self.config.get("retriever", "vector")
        
        if retriever_type == "vector":
            return VectorRetriever(self.embeddings, self.config)
        elif retriever_type == "graph":
            if self.graph is None:
                logger.warning("Graph not available. Falling back to vector retriever.")
                return VectorRetriever(self.embeddings, self.config)
            return GraphRetriever(self.embeddings, self.config)
        elif retriever_type == "path":
            if self.graph is None:
                logger.warning("Graph not available. Falling back to vector retriever.")
                return VectorRetriever(self.embeddings, self.config)
            return PathRetriever(self.embeddings, self.config)
        elif retriever_type == "exact":
            return ExactRetriever(self.embeddings, self.config)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

    def generate(self, query: str, **kwargs) -> str:
        """
        Generate a response for a query.

        Args:
            query: Query string
            **kwargs: Additional arguments for retrieval and generation

        Returns:
            Generated response
        """
        # Retrieve context
        context = self.retriever.retrieve(query, **kwargs)
        
        # Generate response
        response = self.generator.generate(query, context)
        
        return response
    
    def generate_with_citations(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a response with citations.

        Args:
            query: Query string
            **kwargs: Additional arguments for retrieval and generation

        Returns:
            Dictionary with response and citations
        """
        if self.citation is None:
            raise ValueError("Citation not enabled. Configure citation in the pipeline.")
        
        # Retrieve context
        context = self.retriever.retrieve(query, **kwargs)
        
        # Generate response
        response = self.generator.generate(query, context)
        
        # Find citations
        citations = self.citation.find_sources(response, context)
        
        # Verify response
        verification = self.citation.verify_response(response, context)
        
        return {
            "response": response,
            "citations": citations,
            "verification": verification
        }
    
    def evaluate(self, query: str, response: str, ground_truth: str) -> Dict[str, float]:
        """
        Evaluate the quality of a response.

        Args:
            query: Query string
            response: Generated response
            ground_truth: Ground truth response

        Returns:
            Evaluation metrics
        """
        # Calculate semantic similarity
        similarity = float(self.embeddings.similarity(response, ground_truth))
        
        # Calculate token overlap (simple metric)
        response_tokens = set(response.lower().split())
        ground_truth_tokens = set(ground_truth.lower().split())
        
        if ground_truth_tokens:
            precision = len(response_tokens.intersection(ground_truth_tokens)) / len(response_tokens) if response_tokens else 0
            recall = len(response_tokens.intersection(ground_truth_tokens)) / len(ground_truth_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1 = 0
        
        return {
            "similarity": similarity,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
