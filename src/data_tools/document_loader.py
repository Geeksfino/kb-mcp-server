#!/usr/bin/env python3
"""Command line tool for loading documents into txtai database."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from .loader_utils import DocumentLoader
from .config import LoaderConfig, ProcessorConfig, EmbeddingsConfig, DatabaseConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="""
Document Loader for txtai Embeddings Database

This tool loads documents into a txtai embeddings database, supporting various storage backends
and processing options. Documents are chunked, embedded, and stored for later semantic search.

Supported File Types:
  - Text files (.txt)
  - Markdown files (.md)
  - PDF documents (.pdf)
  - Microsoft Word documents (.doc, .docx)

Examples:
  # Basic usage with default SQLite storage
  %(prog)s --input docs/ --model sentence-transformers/all-MiniLM-L6-v2

  # Use PostgreSQL with pgvector for both content and vector storage
  %(prog)s --input docs/ \\
    --vector-backend pgvector \\
    --vector-url "postgresql://user:pass@localhost:5432/dbname" \\
    --content-url "postgresql://user:pass@localhost:5432/dbname"

  # Process documents with GPU acceleration and custom chunk size
  %(prog)s --input docs/ --gpu --chunk-size 2048 --overlap 20

  # Process only specific file types
  %(prog)s --input docs/ --file-types .txt,.md,.pdf
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/Output options
    io_group = parser.add_argument_group('Input/Output Options')
    io_group.add_argument(
        '--input',
        required=True,
        help='Input file or directory path to process'
    )
    io_group.add_argument(
        '--file-types',
        help='Comma-separated list of file extensions to process (default: .txt,.md,.pdf,.doc,.docx)'
    )
    io_group.add_argument(
        '--index-path',
        help='Path to save/load the embeddings index'
    )
    io_group.add_argument(
        '--read-only',
        action='store_true',
        help='Open database in read-only mode'
    )
    io_group.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively process subdirectories'
    )
    io_group.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Content Storage options
    content_group = parser.add_argument_group('Content Storage Options')
    content_group.add_argument(
        '--content-url',
        help='Database URL for content storage (uses SQLite if not provided)'
    )
    content_group.add_argument(
        '--content-schema',
        help='Database schema for content storage'
    )
    
    # Vector Storage options
    vector_group = parser.add_argument_group('Vector Storage Options')
    vector_group.add_argument(
        '--vector-backend',
        choices=['faiss', 'pgvector', 'sqlite-vec'],
        default='faiss',
        help='Vector storage backend (default: faiss)'
    )
    vector_group.add_argument(
        '--vector-url',
        help='Database URL for pgvector (required if using pgvector)'
    )
    vector_group.add_argument(
        '--vector-schema',
        help='Database schema for pgvector'
    )
    vector_group.add_argument(
        '--vector-table',
        default='vectors',
        help='Table name for vector storage (default: vectors)'
    )
    
    # Document Processing options
    proc_group = parser.add_argument_group('Document Processing Options')
    proc_group.add_argument(
        '--chunk-size',
        type=int,
        default=2048,
        help='Size of text chunks in characters (default: 2048)'
    )
    proc_group.add_argument(
        '--overlap',
        type=int,
        default=20,
        help='Overlap between chunks in characters (default: 20)'
    )
    proc_group.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    
    # Model options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument(
        '--model',
        default="sentence-transformers/all-MiniLM-L6-v2",
        help='Sentence transformer model name (default: sentence-transformers/all-MiniLM-L6-v2)'
    )
    model_group.add_argument(
        '--similarity',
        choices=['cosine', 'l2'],
        default='cosine',
        help='Similarity metric (default: cosine)'
    )
    model_group.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for model inference'
    )

    return parser.parse_args()

def progress_callback(total_processed: int):
    """Callback function to report progress."""
    logger.info(f"Processed {total_processed} documents")

def main():
    """Main entry point for the document loader."""
    args = parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Validate arguments
        if args.vector_backend == 'pgvector' and not args.vector_url:
            raise ValueError("--vector-url is required when using pgvector backend")
        
        # Parse file types if provided
        supported_extensions = ['.txt', '.md', '.pdf', '.doc', '.docx']
        if args.file_types:
            supported_extensions = [
                ext if ext.startswith('.') else f'.{ext}'
                for ext in args.file_types.split(',')
            ]
        
        # Create database configuration
        database_config = DatabaseConfig(
            content_url=args.content_url,
            content_schema=args.content_schema,
            vector_backend=args.vector_backend,
            vector_url=args.vector_url,
            vector_schema=args.vector_schema,
            vector_table=args.vector_table,
            index_path=args.index_path,
            writable=not args.read_only
        )
        
        # Create processor configuration
        processor_config = ProcessorConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
            supported_extensions=supported_extensions
        )
        
        # Create embeddings configuration
        embeddings_config = EmbeddingsConfig(
            model_name=args.model,
            batch_size=args.batch_size,
            similarity=args.similarity,
            backend='faiss',  # default backend
            gpu=args.gpu
        )
        
        # Create loader configuration
        loader_config = LoaderConfig(
            database=database_config,
            processor=processor_config,
            embeddings=embeddings_config,
            batch_size=args.batch_size
        )
        
        # Create loader
        loader = DocumentLoader(loader_config)
        
        # Process input path
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {args.input}")
        
        logger.info(f"Processing documents from: {args.input}")
        chunks = loader.process_documents(
            input_path=str(input_path),
            recursive=args.recursive
        )
        
        # Load chunks into database
        loader.load_to_database(chunks, progress_callback=progress_callback)
        
        # Print final statistics
        stats = loader.get_stats()
        logger.info("Document loading completed successfully")
        logger.info(f"Total documents processed: {stats['total_processed']}")
        logger.info(f"Total documents in database: {stats['total_documents']}")
        logger.info("Database configuration:")
        logger.info(f"  Vector backend: {args.vector_backend}")
        if args.vector_backend == 'pgvector':
            logger.info(f"  Vector URL: {args.vector_url}")
        if args.index_path:
            logger.info(f"  Index path: {args.index_path}")
        logger.info(f"  Model: {stats['config']['model']}")
        logger.info(f"  Backend: {stats['config']['backend']}")
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
