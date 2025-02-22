#!/usr/bin/env fish

# Default values
set -l input_dir ""
set -l model "sentence-transformers/all-MiniLM-L6-v2"
set -l chunk_size 2048
set -l overlap 20
set -l batch_size 32
set -l file_types ".txt,.md,.pdf,.doc,.docx"
set -l use_gpu false
set -l recursive false
set -l verbose false

# Database settings
set -l db_backend "faiss"  # faiss, pgvector, sqlite-vec
set -l db_url ""
set -l content_url ""

function show_help
    echo "Usage: "(status basename)" [options]

Load documents into txtai embeddings database.

Required:
    -i, --input DIR         Input directory containing documents

Optional:
    -m, --model NAME        Model name (default: sentence-transformers/all-MiniLM-L6-v2)
    -c, --chunk-size NUM    Chunk size in characters (default: 2048)
    -o, --overlap NUM       Chunk overlap in characters (default: 20)
    -b, --batch-size NUM    Processing batch size (default: 32)
    -f, --file-types LIST   Comma-separated list of extensions (default: .txt,.md,.pdf,.doc,.docx)
    -g, --gpu              Use GPU acceleration
    -r, --recursive        Process subdirectories recursively
    -v, --verbose         Enable verbose logging

Database Options:
    --db-backend TYPE      Database backend: faiss, pgvector, sqlite-vec (default: faiss)
    --db-url URL          Database URL for pgvector
    --content-url URL     Database URL for content storage

Examples:
    # Basic usage with default settings
    "(status basename)" -i /path/to/docs

    # Use GPU and process recursively
    "(status basename)" -i /path/to/docs -g -r

    # Use PostgreSQL with pgvector
    "(status basename)" -i /path/to/docs \\
      --db-backend pgvector \\
      --db-url \"postgresql://user:pass@localhost:5432/dbname\" \\
      --content-url \"postgresql://user:pass@localhost:5432/dbname\""
end

# Parse arguments
set -l options (fish_opt -s h -l help)
set options $options (fish_opt -s i -l input --required-val)
set options $options (fish_opt -s m -l model --required-val)
set options $options (fish_opt -s c -l chunk-size --required-val)
set options $options (fish_opt -s o -l overlap --required-val)
set options $options (fish_opt -s b -l batch-size --required-val)
set options $options (fish_opt -s f -l file-types --required-val)
set options $options (fish_opt -s g -l gpu)
set options $options (fish_opt -s r -l recursive)
set options $options (fish_opt -s v -l verbose)
set options $options (fish_opt -l db-backend --required-val)
set options $options (fish_opt -l db-url --required-val)
set options $options (fish_opt -l content-url --required-val)

argparse $options -- $argv
or begin
    show_help
    exit 1
end

if set -q _flag_help
    show_help
    exit 0
end

# Required arguments
if not set -q _flag_input
    echo "Error: Input directory is required"
    show_help
    exit 1
end
set input_dir $_flag_input

# Optional arguments
set -q _flag_model; and set model $_flag_model
set -q _flag_chunk_size; and set chunk_size $_flag_chunk_size
set -q _flag_overlap; and set overlap $_flag_overlap
set -q _flag_batch_size; and set batch_size $_flag_batch_size
set -q _flag_file_types; and set file_types $_flag_file_types
set -q _flag_gpu; and set use_gpu true
set -q _flag_recursive; and set recursive true
set -q _flag_verbose; and set verbose true
set -q _flag_db_backend; and set db_backend $_flag_db_backend
set -q _flag_db_url; and set db_url $_flag_db_url
set -q _flag_content_url; and set content_url $_flag_content_url

# Build command
set -l cmd "python -m src.data_tools.document_loader"
set cmd "$cmd --input \"$input_dir\""
set cmd "$cmd --model \"$model\""
set cmd "$cmd --chunk-size $chunk_size"
set cmd "$cmd --overlap $overlap"
set cmd "$cmd --batch-size $batch_size"
set cmd "$cmd --file-types \"$file_types\""

# Add optional flags
test "$use_gpu" = true; and set cmd "$cmd --gpu"
test "$recursive" = true; and set cmd "$cmd --recursive"
test "$verbose" = true; and set cmd "$cmd --verbose"

# Add database options
set cmd "$cmd --vector-backend \"$db_backend\""
test -n "$db_url"; and set cmd "$cmd --vector-url \"$db_url\""
test -n "$content_url"; and set cmd "$cmd --content-url \"$content_url\""

# Print command if verbose
test "$verbose" = true; and echo "Running: $cmd"

# Execute command
eval $cmd
