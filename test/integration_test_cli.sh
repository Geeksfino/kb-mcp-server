#!/bin/bash
#
# Integration test for the data tools CLI functionality
#
# This script tests the complete workflow of:
# 1. Document ingestion
# 2. Knowledge graph building
# 3. Search functionality (both standard and graph-enhanced)
#
# It creates a temporary test environment with configuration and test documents,
# then runs the CLI commands to test each step of the workflow.

set -e

echo "=== Testing Data Tools CLI Functionality ==="

# Create a temporary directory for test files
TEST_DIR=$(mktemp -d)
echo "Created test directory: $TEST_DIR"

# Create a test configuration file
CONFIG_FILE="$TEST_DIR/config.yaml"
cat > "$CONFIG_FILE" << EOF
path: $TEST_DIR/knowledge-base
embeddings:
  path: sentence-transformers/all-MiniLM-L6-v2
  storagepath: $TEST_DIR/embeddings
graph:
  similarity: 0.6
  limit: 10
EOF
echo "Created test configuration file: $CONFIG_FILE"

# Create a test document
TEST_DOCUMENT="$TEST_DIR/test_document.txt"
cat > "$TEST_DOCUMENT" << EOF
This is a test document about machine learning.
Machine learning is a field of artificial intelligence that uses statistical techniques to give
computer systems the ability to "learn" from data, without being explicitly programmed.
The name machine learning was coined in 1959 by Arthur Samuel.
EOF
echo "Created test document: $TEST_DOCUMENT"

# Set environment variable for configuration
export KNOWLEDGE_BASE_CONFIG="$CONFIG_FILE"
echo "Set KNOWLEDGE_BASE_CONFIG environment variable to: $CONFIG_FILE"
echo

# Test document ingestion
echo "=== Testing Document Ingestion ==="
python -m src.data_tools.cli ingest file "$TEST_DOCUMENT"
if [ $? -eq 0 ]; then
    echo "✓ Document ingestion successful"
else
    echo "✗ Document ingestion failed"
    exit 1
fi
echo

# Wait a bit to ensure the database is ready
sleep 10

# Test knowledge graph building
echo "=== Testing Knowledge Graph Building ==="
python -m src.data_tools.cli graph --min-similarity 0.6
if [ $? -eq 0 ]; then
    echo "✓ Knowledge graph building successful"
else
    echo "✗ Knowledge graph building failed"
    exit 1
fi
echo

# Wait a bit to ensure the graph is ready
sleep 10

# Test search functionality
echo "=== Testing Search Functionality ==="
python -m src.data_tools.cli search "What is machine learning?" --limit 3
if [ $? -eq 0 ]; then
    echo "✓ Standard search successful"
else
    echo "✗ Standard search failed"
    exit 1
fi
echo

# Test search with knowledge graph
echo "=== Testing Search with Knowledge Graph ==="
python -m src.data_tools.cli search "What is machine learning?" --limit 3 --use-graph
if [ $? -eq 0 ]; then
    echo "✓ Graph search successful"
else
    echo "✗ Graph search failed"
    exit 1
fi
echo

echo "=== All tests completed successfully ==="
echo "Test files are in $TEST_DIR"
echo "You can remove the test directory with: rm -rf $TEST_DIR"
