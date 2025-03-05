#!/usr/bin/env python
"""
Configuration Helper for txtai Knowledge Base

This script helps users select and customize domain-specific configuration templates.
"""

import os
import sys
import yaml
import shutil
import argparse
from pathlib import Path

# Get the directory where templates are stored
TEMPLATE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

def list_templates():
    """List all available templates."""
    templates = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith('.yml')]
    print("\nAvailable templates:")
    for template in templates:
        print(f"  - {template}")
    print()

def view_template(template_name):
    """View the contents of a template."""
    template_path = TEMPLATE_DIR / template_name
    if not template_path.exists():
        print(f"Template {template_name} not found.")
        return
    
    with open(template_path, 'r') as f:
        content = f.read()
    
    print(f"\nContents of {template_name}:")
    print("-" * 50)
    print(content)
    print("-" * 50)

def create_custom_config(template_name, output_path, **kwargs):
    """Create a custom configuration based on a template."""
    template_path = TEMPLATE_DIR / template_name
    if not template_path.exists():
        print(f"Template {template_name} not found.")
        return
    
    # Load template
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply customizations
    if 'path' in kwargs and kwargs['path']:
        config['path'] = kwargs['path']
    
    if 'model' in kwargs and kwargs['model']:
        config['embeddings']['path'] = kwargs['model']
    
    if 'max_hops' in kwargs and kwargs['max_hops'] is not None:
        config['embeddings']['graph']['search']['max_hops'] = kwargs['max_hops']
    
    if 'min_score' in kwargs and kwargs['min_score'] is not None:
        config['embeddings']['graph']['search']['min_score'] = kwargs['min_score']
    
    if 'limit' in kwargs and kwargs['limit'] is not None:
        config['search']['limit'] = kwargs['limit']
    
    # Save to output path
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\nCustom configuration created at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Configuration helper for txtai knowledge base")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available templates')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View a template')
    view_parser.add_argument('template', help='Template name (e.g., data_science.yml)')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a custom configuration')
    create_parser.add_argument('template', help='Base template name (e.g., data_science.yml)')
    create_parser.add_argument('output', help='Output file path')
    create_parser.add_argument('--path', help='Knowledge base path')
    create_parser.add_argument('--model', help='Embedding model path')
    create_parser.add_argument('--max-hops', type=int, help='Maximum graph hops')
    create_parser.add_argument('--min-score', type=float, help='Minimum score threshold')
    create_parser.add_argument('--limit', type=int, help='Result limit')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_templates()
    elif args.command == 'view':
        view_template(args.template)
    elif args.command == 'create':
        create_custom_config(
            args.template, 
            args.output,
            path=args.path,
            model=args.model,
            max_hops=args.max_hops,
            min_score=args.min_score,
            limit=args.limit
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
