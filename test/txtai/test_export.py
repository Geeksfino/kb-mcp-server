#!/usr/bin/env python3
"""
Test script to demonstrate the export functionality of the CLI.
"""

import os
import tempfile
import subprocess
from pathlib import Path

def main():
    # Create a temporary directory for the exported index
    with tempfile.TemporaryDirectory() as temp_dir:
        export_path = os.path.join(temp_dir, "test_index")
        
        # Path to the document
        doc_path = os.path.abspath("./test/knowledgebase/data_science.md")
        
        # Path to the configuration file
        config_path = os.path.abspath("./test/txtai/kb.yml")
        
        # Make sure we're in the right directory
        current_dir = os.getcwd()
        if not current_dir.endswith('embedding-mcp-server'):
            print(f"Warning: Current directory is {current_dir}")
            print("This script should be run from the embedding-mcp-server directory")
        
        print(f"Using document: {doc_path}")
        print(f"Using configuration: {config_path}")
        print(f"Export path: {export_path}.tar.gz")
        
        # Build the command
        cmd = [
            "python", "-m", "data_tools.cli", 
            "build", 
            "--input", doc_path, 
            "--config", config_path,
            "--export", export_path
        ]
        
        # Run the command
        print("\nRunning command:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print the output
        print("\nCommand output:")
        print(result.stdout)
        
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
        
        # Check if the export file was created
        export_file = f"{export_path}.tar.gz"
        if os.path.exists(export_file):
            print(f"\nExport successful! File created at: {export_file}")
            print(f"File size: {os.path.getsize(export_file) / 1024:.2f} KB")
            
            # Test loading the exported index
            print("\nTesting loading the exported index...")
            
            # Create a simple script to load the index
            load_script = f"""
from txtai.app import Application
app = Application("path: {export_file}")
results = app.search("What is data science?", limit=1)
print(f"Search results: {{results}}")
            """
            
            # Create a temporary Python file
            load_script_path = os.path.join(temp_dir, "load_test.py")
            with open(load_script_path, "w") as f:
                f.write(load_script)
            
            # Run the script
            load_cmd = ["python", load_script_path]
            load_result = subprocess.run(load_cmd, capture_output=True, text=True)
            
            print("\nLoad test output:")
            print(load_result.stdout)
            
            if load_result.stderr:
                print("\nLoad test errors:")
                print(load_result.stderr)
        else:
            print(f"\nExport failed! File not found at: {export_file}")

if __name__ == "__main__":
    main()
