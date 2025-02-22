# Data Tools for txtai

This package provides tools for loading and processing documents into txtai databases.

## Document Loader

The document loader is a command-line tool that helps you load various document types into a txtai database. It supports:

- PDF files (.pdf)
- Word documents (.doc, .docx)
- Text files (.txt)
- Markdown files (.md)

### Features

- Automatic text extraction from supported document types
- Smart text chunking with configurable size and overlap
- Batch processing for efficient database loading
- Support for recursive directory processing
- Progress tracking and error handling

### Usage

```bash
python -m data_tools.document_loader --input <path> --db-url <url> [options]

Required arguments:
  --input PATH           Input file or directory path
  --db-url URL          URL of the txtai database

Optional arguments:
  --chunk-size SIZE     Size of text chunks (default: 512)
  --overlap SIZE        Overlap between chunks (default: 50)
  --batch-size SIZE     Batch size for processing (default: 32)
  --recursive           Recursively process directories
  --verbose            Enable verbose logging
```

### Examples

1. Load a single PDF file:
```bash
python -m data_tools.document_loader --input document.pdf --db-url http://localhost:8000
```

2. Process a directory of documents recursively:
```bash
python -m data_tools.document_loader --input ./documents --db-url http://localhost:8000 --recursive
```

3. Customize chunking parameters:
```bash
python -m data_tools.document_loader --input ./documents --db-url http://localhost:8000 --chunk-size 1024 --overlap 100
```

## Development

The tool is built using txtai's pipeline and workflow systems. Key components:

- `document_loader.py`: Command-line interface
- `loader_utils.py`: Core document processing functionality

### Adding Support for New Document Types

To add support for new document types, update the `SUPPORTED_EXTENSIONS` set in the `DocumentProcessor` class.
