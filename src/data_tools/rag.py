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
from txtai.app import Application

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


class TxtaiRetriever(Retriever):
    """Unified retriever that leverages txtai's built-in search capabilities."""

    def __init__(self, embeddings: Embeddings, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TxtaiRetriever.

        Args:
            embeddings: txtai Embeddings instance
            config: Configuration dictionary
        """
        self.embeddings = embeddings
        self.config = config or {}
        
        # Configuration parameters
        self.limit = self.config.get("limit", 10)
        self.min_score = self.config.get("min_score", 0.3)
        self.search_type = self.config.get("search_type", "vector")
        self.max_hops = self.config.get("max_hops", 2)
    
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query using txtai's built-in search capabilities.

        Args:
            query: Query string
            **kwargs: Additional arguments for retrieval
                - limit: Maximum number of results to return
                - min_score: Minimum similarity score
                - search_type: Type of search to perform (vector, graph, hybrid, exact)
                - max_hops: Maximum number of hops for graph search

        Returns:
            List of context documents
        """
        # Override config with kwargs if provided
        limit = kwargs.get("limit", self.limit)
        min_score = kwargs.get("min_score", self.min_score)
        search_type = kwargs.get("search_type", self.search_type)
        max_hops = kwargs.get("max_hops", self.max_hops)
        
        # Build SQL-like query based on search type
        sql_query = "select id, text, score"
        
        # Add title to select if available in schema
        try:
            if "title" in self.embeddings.config.get("columns", {}):
                sql_query += ", title"
        except:
            pass
        
        sql_query += " from txtai where "
        
        # Different search types
        if search_type == "vector":
            sql_query += f"similar('{query}')"
        elif search_type == "graph" and hasattr(self.embeddings, 'graph') and self.embeddings.graph:
            sql_query += f"graph('{query}', {max_hops}, {limit*2})"
        elif search_type == "hybrid" and hasattr(self.embeddings, 'graph') and self.embeddings.graph:
            # Combine vector and graph search
            sql_query += f"(similar('{query}') or graph('{query}', {max_hops}, {limit}))"
        elif search_type == "exact":
            # Escape single quotes in query
            escaped_query = query.replace("'", "''")
            sql_query += f"text ~ '{escaped_query}'"
        else:
            # Default to vector search
            sql_query += f"similar('{query}')"
        
        # Add limit
        sql_query += f" limit {limit}"
        
        try:
            # Execute search
            results = self.embeddings.search(sql_query)
            
            # Filter by score
            results = [r for r in results if r.get("score", 0) >= min_score]
            
            # Ensure consistent format
            for result in results:
                # Ensure score is a float
                if "score" in result:
                    result["score"] = float(result["score"])
            
            return results
            
        except Exception as e:
            logger.warning(f"Search failed: {str(e)}. Falling back to basic vector search.")
            # Fall back to basic vector search
            try:
                return self.embeddings.search(query, limit=limit)
            except Exception as e2:
                logger.error(f"Fallback search also failed: {str(e2)}")
                return []


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

    def generate(self, query: str, context: List[Dict[str, Any]], **kwargs) -> str:
        """
        Generate a response using an LLM.

        Args:
            query: Query string
            context: Context documents
            **kwargs: Additional arguments for generation

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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Citation.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Configuration parameters
        self.min_similarity = self.config.get("min_similarity", 0.7)
        
        # Import nltk only when needed
        try:
            import nltk
            self.nltk = nltk
            
            # Download required nltk resources
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
        except ImportError:
            logger.warning("nltk not available. Citation will not work properly.")
            self.nltk = None

    def find_sources(self, response: str, context: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find sources for each sentence in the response.

        Args:
            response: Generated response
            context: Context documents

        Returns:
            Dictionary mapping sentences to source documents
        """
        if self.nltk is None:
            logger.warning("nltk not available. Cannot find sources.")
            return {}
        
        # Tokenize response into sentences
        sentences = self.nltk.sent_tokenize(response)
        
        # Find sources for each sentence
        sources = {}
        for sentence in sentences:
            sentence_sources = []
            
            for doc in context:
                if "text" not in doc:
                    continue
                
                # Calculate similarity between sentence and document
                try:
                    # Use cosine similarity if available
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.metrics.pairwise import cosine_similarity
                    
                    vectorizer = TfidfVectorizer().fit_transform([sentence, doc["text"]])
                    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
                except:
                    # Fall back to simple string matching
                    similarity = 0
                    if sentence.lower() in doc["text"].lower():
                        similarity = 0.9
                
                # Add document as source if similarity is high enough
                if similarity >= self.min_similarity:
                    source = {
                        "id": doc.get("id", ""),
                        "title": doc.get("title", ""),
                        "text": doc.get("text", ""),
                        "similarity": similarity
                    }
                    sentence_sources.append(source)
            
            if sentence_sources:
                # Sort sources by similarity
                sentence_sources.sort(key=lambda x: x["similarity"], reverse=True)
                sources[sentence] = sentence_sources
        
        return sources

    def add_citations(self, response: str, context: List[Dict[str, Any]]) -> str:
        """
        Add citations to the response.

        Args:
            response: Generated response
            context: Context documents

        Returns:
            Response with citations
        """
        if self.nltk is None:
            logger.warning("nltk not available. Cannot add citations.")
            return response
        
        # Find sources
        sources = self.find_sources(response, context)
        
        if not sources:
            return response
        
        # Add footnotes at the end
        footnotes = "\n\nSources:\n"
        source_map = {}  # Map sources to numbers
        source_counter = 1
        
        # Add citations to the response
        cited_response = ""
        for sentence in self.nltk.sent_tokenize(response):
            if sentence in sources:
                # Get source IDs
                citation_ids = []
                for source in sources[sentence]:
                    source_id = source.get("id", "") or source.get("title", "")
                    if source_id:
                        if source_id not in source_map:
                            source_map[source_id] = source_counter
                            
                            # Add footnote
                            title = source.get("title", "")
                            footnotes += f"[{source_counter}] {title or source_id}\n"
                            source_counter += 1
                        
                        citation_ids.append(str(source_map[source_id]))
                
                # Add citation to sentence
                if citation_ids:
                    cited_response += sentence + " [" + ", ".join(citation_ids) + "] "
                else:
                    cited_response += sentence + " "
            else:
                cited_response += sentence + " "
        
        # Add footnotes if we have any
        if len(source_map) > 0:
            return cited_response.strip() + footnotes
        else:
            return cited_response.strip()

    def verify_response(self, response: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify that the response is supported by the context.

        Args:
            response: Generated response
            context: Context documents

        Returns:
            Verification results
        """
        if self.nltk is None:
            logger.warning("nltk not available. Cannot verify response.")
            return {"verified": False, "coverage": 0}
        
        # Find sources
        sources = self.find_sources(response, context)
        
        # Calculate coverage
        sentences = self.nltk.sent_tokenize(response)
        coverage = len(sources) / len(sentences) if sentences else 0
        
        return {
            "coverage": coverage,
            "verified": coverage >= 0.7  # Consider verified if 70% of sentences have sources
        }

class RAGPipeline:
    """
    Retrieval Augmented Generation (RAG) pipeline.
    
    This class coordinates the RAG process:
    1. Retrieves relevant context for a query
    2. Generates a response using the retrieved context
    3. Optionally adds citations to the response
    """

    def __init__(self, embeddings: Embeddings, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG pipeline.

        Args:
            embeddings: txtai Embeddings instance
            config: Configuration dictionary
        """
        self.embeddings = embeddings
        self.config = config or {}
        
        # Create components
        self.retriever = TxtaiRetriever(embeddings, self.config.get("retriever", {}))
        self.generator = Generator(self.config.get("generator", {}))
        
        # Create citation handler if enabled
        citation_config = self.config.get("citation", {})
        if citation_config.get("enabled", False):
            self.citation = Citation(citation_config)
        else:
            self.citation = None

    def generate(self, query: str, **kwargs) -> str:
        """
        Generate a response to a query using RAG.

        Args:
            query: Query string
            **kwargs: Additional arguments for generation
                - limit: Maximum number of context documents to retrieve
                - search_type: Type of search to perform (vector, graph, hybrid, exact)
                - max_hops: Maximum number of hops for graph search
                - min_score: Minimum similarity score for retrieved documents
                - prompt_template: Custom prompt template
                - system_prompt: Custom system prompt
                - add_citations: Whether to add citations to the response

        Returns:
            Generated response
        """
        # Get retrieval parameters
        limit = kwargs.get("limit", self.config.get("limit", 5))
        search_type = kwargs.get("search_type", self.config.get("search_type", "vector"))
        max_hops = kwargs.get("max_hops", self.config.get("max_hops", 2))
        min_score = kwargs.get("min_score", self.config.get("min_score", 0.3))
        
        # Retrieve context
        context = self.retriever.retrieve(
            query, 
            limit=limit,
            search_type=search_type,
            max_hops=max_hops,
            min_score=min_score
        )
        
        if not context:
            logger.warning("No context found for query")
            # Generate response without context
            return self.generator.generate(query, context=[], **kwargs)
        
        # Generate response with context
        response = self.generator.generate(query, context=context, **kwargs)
        
        # Add citations if enabled
        if self.citation and kwargs.get("add_citations", self.config.get("add_citations", True)):
            response = self.citation.add_citations(response, context)
        
        return response

    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Retrieve context for a query.

        Args:
            query: Query string
            **kwargs: Additional arguments for retrieval

        Returns:
            List of context documents
        """
        return self.retriever.retrieve(query, **kwargs)

# Graph utility functions
def extract_facts_from_graph(graph, query=None, num_facts=5, llm=None):
    """
    Extract facts from a graph using a language model.
    
    Args:
        graph: txtai graph object
        query: Optional query to filter nodes
        num_facts: Number of facts to extract
        llm: Optional language model instance
        
    Returns:
        List of extracted facts
    """
    # If query is provided, use it to filter the graph
    if query:
        # Use search to find relevant nodes
        try:
            filtered_graph = graph.search(query, 10)
            nodes = list(filtered_graph.centrality().keys())[:10]
        except Exception as e:
            logger.warning(f"Error searching graph: {e}")
            # Fall back to centrality
            nodes = list(graph.centrality().keys())[:10]
    else:
        # Otherwise use centrality to get the most important nodes
        nodes = list(graph.centrality().keys())[:10]
    
    # Extract text from the selected nodes
    texts = []
    for node_id in nodes:
        try:
            node = graph.node(node_id)
            if node and "text" in node:
                texts.append(node["text"])
        except Exception as e:
            logger.debug(f"Error getting node {node_id}: {e}")
    
    text = "\n".join(texts)
    
    if not text:
        return ["No text found in graph nodes"]
    
    # If we have an LLM configured, use it to extract facts
    try:
        if llm is None:
            from txtai import LLM
            llm = LLM()
        
        # Create the prompt for the LLM
        prompt = f"""Extract {num_facts} key facts from the following text. 
        Provide each fact as a concise, standalone statement.
        
        Text:
        {text}
        
        Facts:
        """
        
        # Get the response from the LLM
        response = llm(prompt)
        
        # Parse the response into a list of facts
        facts = []
        for line in response.split('\n'):
            line = line.strip()
            # Skip empty lines and lines that don't look like facts
            if not line or len(line) < 10:
                continue
            # Remove leading numbers and dots if present
            if line[0].isdigit() and line[1:3] in ['. ', '- ', ') ']:
                line = line[3:].strip()
            facts.append(line)
            
        return facts[:num_facts]  # Ensure we don't return more than requested
        
    except Exception as e:
        logger.warning(f"Could not extract facts using LLM: {e}")
        # Fallback: return the first few sentences from the text
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20][:num_facts]

def find_relationship(graph, concept1, concept2, max_path_length=5, llm=None):
    """
    Find and explain the relationship between two concepts in the graph.
    
    Args:
        graph: txtai graph object
        concept1: First concept
        concept2: Second concept
        max_path_length: Maximum path length
        llm: Optional language model instance
        
    Returns:
        Dictionary with relationship information
    """
    # Find nodes related to each concept
    try:
        nodes1 = graph.search(concept1, 3)
        nodes2 = graph.search(concept2, 3)
        
        # Get the IDs of the top nodes for each concept
        id1 = list(nodes1.centrality().keys())[0] if nodes1.centrality() else None
        id2 = list(nodes2.centrality().keys())[0] if nodes2.centrality() else None
    except Exception as e:
        logger.warning(f"Error searching for concepts: {e}")
        return {"error": f"Could not find nodes for concepts: {e}"}
    
    if not id1 or not id2:
        return {"error": f"One or both concepts not found in the graph"}
    
    # Find the shortest path between the two nodes
    try:
        path = graph.showpath(id1, id2, maxlength=max_path_length)
    except Exception as e:
        logger.warning(f"Error finding path: {e}")
        return {"error": f"Error finding path between concepts: {e}"}
    
    if not path:
        return {"error": f"No path found between {concept1} and {concept2} within {max_path_length} steps"}
    
    # Get the text for each node in the path
    path_texts = []
    path_nodes = []
    for node_id in path:
        try:
            node = graph.node(node_id)
            if node:
                path_nodes.append(node)
                if "text" in node:
                    path_texts.append(node["text"])
        except Exception as e:
            logger.debug(f"Error getting node {node_id}: {e}")
    
    # Create a prompt to explain the relationship
    text = "\n".join(path_texts)
    
    explanation = ""
    if text and llm:
        try:
            prompt = f"""Based on the following text, explain the relationship between {concept1} and {concept2}.
            Only use information from the provided text.
            
            {text}"""
            
            explanation = llm(prompt)
        except Exception as e:
            logger.warning(f"Error generating explanation: {e}")
            explanation = f"Path found between {concept1} and {concept2} with {len(path)} nodes."
    else:
        explanation = f"Path found between {concept1} and {concept2} with {len(path)} nodes."
    
    return {
        "path": path,
        "path_nodes": path_nodes,
        "explanation": explanation,
        "concepts": [concept1, concept2]
    }

def generate_questions_from_graph(graph, num_nodes=10, num_questions=5, llm=None):
    """
    Generate interesting questions that could be answered using the graph.
    
    Args:
        graph: txtai graph object
        num_nodes: Number of central nodes to include
        num_questions: Number of questions to generate
        llm: Optional language model instance
        
    Returns:
        List of generated questions
    """
    # Get the most central nodes
    nodes = list(graph.centrality().keys())[:num_nodes]
    
    # Extract text from the selected nodes
    texts = []
    for node_id in nodes:
        try:
            node = graph.node(node_id)
            if node and "text" in node:
                texts.append(node["text"])
        except Exception as e:
            logger.debug(f"Error getting node {node_id}: {e}")
    
    text = "\n".join(texts)
    
    if not text:
        return ["No text found in graph nodes"]
    
    # If we have an LLM, use it to generate questions
    try:
        if llm is None:
            from txtai.pipeline import LLM
            llm = LLM()
        
        # Create the prompt for the LLM
        prompt = f"""Based on the following text, generate {num_questions} interesting questions that could be answered using this information.
        Make the questions diverse and thought-provoking.
        
        {text}"""
        
        # Get the response from the LLM
        response = llm(prompt)
        
        # Parse the response into a list of questions
        questions = []
        for line in response.split('\n'):
            line = line.strip()
            # Only include lines that look like questions
            if '?' in line:
                # Remove leading numbers if present
                if line[0].isdigit() and line[1:3] in ['. ', '- ', ') ']:
                    line = line[3:].strip()
                questions.append(line)
        
        return questions[:num_questions]
        
    except Exception as e:
        logger.warning(f"Could not generate questions using LLM: {e}")
        return [f"Could not generate questions: {e}"]
