"""
Core MCP server implementation for txtai.
"""
import logging
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any
from pathlib import Path

import torch
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.lowlevel.server import Server as MCPServer
from txtai.embeddings import Embeddings

from .config import TxtAISettings
from .context import TxtAIContext

logger = logging.getLogger(__name__)

class TxtAIServer(MCPServer):
    """Custom MCP server with enhanced logging and context management."""
    
    def __init__(self, name: str, instructions: str | None = None):
        logger.debug("Initializing TxtAIServer...")
        super().__init__(name, instructions)
        self._lifespan_context = None
        logger.debug("TxtAIServer initialized")

    @property
    def lifespan_context(self):
        return self._lifespan_context

    @lifespan_context.setter
    def lifespan_context(self, context):
        logger.debug(f"Setting lifespan context: {context}")
        self._lifespan_context = context

    async def handle_message(self, message):
        """Handle messages with detailed logging."""
        logger.debug(f"Received message: {message}")
        try:
            result = await super().handle_message(message)
            logger.debug(f"Message handled successfully: {result}")
            return result
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

@asynccontextmanager
async def txtai_lifespan(ctx: Context) -> AsyncIterator[Dict[str, Any]]:
    """Manage txtai components lifecycle."""
    logger.info("=== Starting txtai server initialization ===")
    try:
        t0 = time.time()
        
        # Load settings from .env
        settings = TxtAISettings.load()
        logger.info(f"Storage mode: {settings.storage_mode}")
        
        # Create embeddings instance
        embeddings_config = settings.get_embeddings_config()
        embeddings_config["content"] = True  # Enable content storage
        embeddings = Embeddings(embeddings_config)
        
        try:
            if settings.storage_mode == "persistence" and Path(settings.index_path).expanduser().exists():
                logger.info(f"Loading existing embeddings from {settings.index_path}")
                embeddings.load(settings.index_path)
                
            elif settings.dataset_enabled and settings.dataset_name:
                # Load from HuggingFace dataset if configured
                try:
                    from datasets import load_dataset
                    logger.info(f"Loading dataset: {settings.dataset_name} (split: {settings.dataset_split})")
                    dataset = load_dataset(settings.dataset_name, split=settings.dataset_split)
                    
                    # Debug dataset structure
                    logger.info(f"Dataset info: {dataset}")
                    logger.info(f"Dataset features: {dataset.features}")
                    logger.info(f"Dataset length: {len(dataset)}")
                    
                    # Clear existing index
                    logger.info("Clearing existing index...")
                    embeddings.index([])
                    
                    # Process all items
                    documents = []
                    for idx, item in enumerate(dataset):
                        try:
                            # Handle SQuAD format
                            if settings.dataset_name == "squad":
                                context = item["context"].strip()
                                question = item["question"].strip()
                                answers = item["answers"]
                                answer_texts = answers["text"]
                                
                                # Store as metadata
                                metadata = {
                                    "text": question,  # Index on question
                                    "context": context,
                                    "answer": answer_texts[0] if answer_texts else "",
                                    "title": item.get("title", "")
                                }
                                
                                doc_id = f"squad_{idx}"
                                documents.append((doc_id, metadata, None))
                                
                            # Handle web_questions format
                            elif settings.dataset_name == "web_questions":
                                metadata = {
                                    "text": item["question"],
                                    "answer": ", ".join(item["answers"])
                                }
                                doc_id = f"webq_{idx}"
                                documents.append((doc_id, metadata, None))
                                
                            if idx < 5:  # Log first 5 documents
                                logger.info(f"Sample document {idx + 1}:")
                                logger.info(f"  ID: {doc_id}")
                                logger.info(f"  Content: {metadata}")
                                
                        except Exception as e:
                            logger.warning(f"Error processing dataset item: {e}")
                            logger.warning(f"Item causing error: {item}")
                            continue
                    
                    if documents:
                        logger.info(f"Indexing {len(documents)} documents...")
                        try:
                            # Index in smaller batches
                            batch_size = 10
                            for i in range(0, len(documents), batch_size):
                                batch = documents[i:i + batch_size]
                                logger.info(f"Indexing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                                embeddings.index(batch)
                                logger.info(f"Batch {i//batch_size + 1} indexed successfully")
                        except Exception as e:
                            logger.error(f"Error during indexing: {e}")
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            raise
                except ImportError as e:
                    logger.error(f"Failed to import datasets library: {e}")
                    logger.warning("Starting with empty embeddings")
                except Exception as e:
                    logger.error(f"Failed to load dataset: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    logger.warning("Starting with empty embeddings")
            else:
                logger.info("Starting with empty embeddings (memory mode, no dataset configured)")
                
        except Exception as e:
            if embeddings_config["gpu"]:
                logger.warning(f"Failed to initialize embeddings with GPU: {e}. Falling back to CPU.")
                embeddings_config["gpu"] = False
                embeddings = Embeddings(embeddings_config)
                logger.info("Successfully initialized embeddings on CPU")
            else:
                raise
        
        # Create context
        txtai_context = TxtAIContext(embeddings=embeddings)
        logger.info(f"Server initialization completed in {time.time() - t0:.2f}s")
        
        yield {"txtai_context": txtai_context}
        
        # Cleanup
        if settings.storage_mode == "persistence":
            storage_path = Path(settings.index_path).expanduser()
            try:
                logger.info(f"Saving embeddings to {storage_path}")
                storage_path.parent.mkdir(parents=True, exist_ok=True)
                embeddings.save(str(storage_path))
                logger.info("Successfully saved embeddings")
            except Exception as e:
                logger.error(f"Failed to save embeddings: {e}")
                
    except Exception as e:
        logger.error(f"Error in txtai lifespan: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def get_text_from_dataset_item(item, dataset_name):
    """Extract text content from different dataset formats.
    
    Known dataset formats:
    - web_questions: {'question': str, 'answers': List[str]}
    - squad: {'data': List[Dict]} where each dict has {'paragraphs': List[Dict]} 
            and each paragraph has {'context': str, 'qas': List[Dict]}
    - wikipedia: {'text': str, 'title': str}
    """
    # Dataset-specific handling
    if dataset_name == "web_questions":
        return item["question"] + " " + " ".join(item["answers"])
    elif "squad" in dataset_name.lower():
        # SQuAD has a nested structure
        if "context" in item:
            return item["context"]
        elif "data" in item:
            # Handle root level
            texts = []
            for article in item["data"]:
                if "paragraphs" in article:
                    for para in article["paragraphs"]:
                        if "context" in para:
                            texts.append(para["context"])
            return "\n".join(texts) if texts else None
        else:
            logger.warning(f"Unexpected SQuAD item format: {item}")
            return None
    elif "wikipedia" in dataset_name.lower():
        return item["text"]
    
    # Generic handling for unknown datasets
    # Try common field names
    for field in ["text", "content", "context", "document", "body"]:
        if isinstance(item, dict) and field in item:
            return item[field]
        if hasattr(item, field):
            return getattr(item, field)
    
    # If it's a simple string
    if isinstance(item, str):
        return item
        
    # If it's a dict, concatenate all string values
    if isinstance(item, dict):
        text_fields = []
        for value in item.values():
            if isinstance(value, str):
                text_fields.append(value)
        if text_fields:
            return " ".join(text_fields)
            
    logger.warning(f"Could not extract text from dataset item: {item}")
    return None

def create_server() -> FastMCP:
    """Create and configure the MCP server."""
    logger.info("Creating TxtAI MCP server...")
    try:
        # Create server instance
        mcp = FastMCP(
            "TxtAI Server",
            lifespan=txtai_lifespan,
            server_class=TxtAIServer,
            dependencies=["txtai", "torch", "transformers"]
        )
        
        # Register components
        from ..tools import register_search_tools
        from ..resources import register_config_resources, register_model_resources
        from ..prompts import register_search_prompts
        
        logger.debug("Registering tools...")
        register_search_tools(mcp)
        
        logger.debug("Registering resources...")
        register_config_resources(mcp)
        register_model_resources(mcp)
        
        logger.debug("Registering prompts...")
        register_search_prompts(mcp)
        
        logger.info("Server created with name: TxtAI Server")
        return mcp
        
    except Exception as e:
        logger.error(f"Error creating server: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise