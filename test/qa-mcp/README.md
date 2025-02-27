# QA MCP Test

This directory contains scripts for testing the txtai MCP server with a question-answering dataset.

## Files

- `qa.yml` - Configuration file for the txtai embeddings
- `load_web_questions.py` - Script to load the web_questions dataset from Hugging Face into txtai embeddings
- `test_qa_server.py` - Script to test the QA functionality with the MCP server

## Usage

### 1. Load the web_questions dataset

First, load the web_questions dataset into txtai embeddings:

```bash
# Install required dependencies
pip install datasets txtai

# Load the full dataset
python load_web_questions.py --test

# Or load a limited number of questions (e.g., 100)
python load_web_questions.py --limit 100 --test

# Save the embeddings as a tar.gz archive
python load_web_questions.py --limit 100 --save-archive qa_embeddings.tar.gz
```

This will:
- Download the web_questions dataset from Hugging Face
- Create a txtai embeddings index in `.txtai/indexes/qa`
- Index the questions and answers
- Run test queries if `--test` is specified
- Save the embeddings as a tar.gz archive if `--save-archive` is specified

### 2. Test the QA functionality with the MCP server

After loading the dataset, test that the QA functionality works with the MCP server:

```bash
# Test using the configuration file (default)
python test_qa_server.py

# Test using a directory of embeddings
python test_qa_server.py --embeddings .txtai/indexes/qa

# Test using an archive file
python test_qa_server.py --embeddings qa_embeddings.tar.gz
```

This will:
- Load the embeddings from the specified source (config file, directory, or archive)
- Test the answer_question functionality directly

### 3. Run the MCP server with the QA embeddings

Finally, run the MCP server with the QA embeddings:

```bash
cd ../../
# Using a directory
mcp run --transport sse server.py --embeddings .txtai/indexes/qa

# Or using an archive file
mcp run --transport sse server.py --embeddings test/qa-mcp/qa_embeddings.tar.gz
```

This will:
- Start the MCP server
- Load the QA embeddings (from either a directory or an archive file)
- Register the search and QA tools

## Notes

- The web_questions dataset contains about 3,778 question-answer pairs
- The embeddings are stored using SQLite for persistence
- The extractor pipeline uses the `distilbert-base-cased-distilled-squad` model for extractive QA
- When creating a tar.gz archive, always use the `--save-archive` option or the `embeddings.save()` method, as manually archiving the directory will not create the correct internal structure
