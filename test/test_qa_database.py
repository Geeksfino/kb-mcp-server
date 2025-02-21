#!/usr/bin/env python3
"""
QA Database test module using txtai for semantic search and question answering.
Run this module directly using: python test_qa_database.py
"""

import os
import pickle
import time
from datasets import load_dataset
print(f"Time after first imports: {time.time()}")

print("Loading txtai...")
t0 = time.time()
from txtai.embeddings import Embeddings
print(f"Time to import txtai: {time.time() - t0:.2f}s")

# Constants
MODEL_PATH = "sentence-transformers/nli-mpnet-base-v2"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
DATA_CACHE = os.path.join(CACHE_DIR, "data.pkl")
INDEX_CACHE = os.path.join(CACHE_DIR, "index_data.pkl")

def clean_cache():
    """Remove corrupted cache files."""
    if os.path.exists(DATA_CACHE):
        os.remove(DATA_CACHE)
    if os.path.exists(INDEX_CACHE):
        os.remove(INDEX_CACHE)

def load_qa_dataset(num_samples=5, use_cache=True):
    """Load and return sample QA pairs from the web_questions dataset."""
    t0 = time.time()
    if use_cache:
        try:
            if os.path.exists(DATA_CACHE):
                print("Loading dataset from cache...")
                with open(DATA_CACHE, 'rb') as f:
                    ds = pickle.load(f)
                print(f"Time to load dataset from cache: {time.time() - t0:.2f}s")
                return ds
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache corrupted, rebuilding... ({str(e)})")
            clean_cache()

    print("Loading web_questions dataset...")
    ds = load_dataset("web_questions", split="train")
    print(f"Time to load dataset from scratch: {time.time() - t0:.2f}s")
    
    # Print sample questions and answers
    print("\nSample QA pairs:")
    print("-" * 50)
    for row in ds.select(range(num_samples)):
        print(f"Q: {row['question']}")
        print(f"A: {', '.join(row['answers'])}\n")
    
    # Cache the dataset
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(DATA_CACHE, 'wb') as f:
            pickle.dump(ds, f)
    
    return ds

def prepare_index_data(dataset):
    """Prepare data for indexing."""
    return [
        (uid, {
            "text": row["question"],
            "answer": ", ".join(row["answers"])
        }, None)
        for uid, row in enumerate(dataset)
    ]

def build_qa_index(dataset, use_cache=True):
    """Build a txtai index from the dataset."""
    t0_total = time.time()
    
    print("\nInitializing embeddings model...")
    t0 = time.time()
    # Create embeddings index with content storage enabled
    embeddings = Embeddings({
        "path": MODEL_PATH,
        "content": True
    })
    print(f"Time to initialize embeddings model: {time.time() - t0:.2f}s")
    
    if use_cache:
        try:
            if os.path.exists(INDEX_CACHE):
                print("\nLoading index from cache...")
                t0 = time.time()
                with open(INDEX_CACHE, 'rb') as f:
                    index_data = pickle.load(f)
                embeddings.index(index_data)
                print(f"Time to load index from cache: {time.time() - t0:.2f}s")
                return embeddings
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Cache corrupted, rebuilding... ({str(e)})")
            clean_cache()

    print("\nBuilding QA index...")
    t0 = time.time()
    index_data = prepare_index_data(dataset)
    embeddings.index(index_data)
    print(f"Time to build index from scratch: {time.time() - t0:.2f}s")

    # Cache the index data
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_CACHE, 'wb') as f:
            pickle.dump(index_data, f)
    
    print(f"Total time for index building: {time.time() - t0_total:.2f}s")
    return embeddings

def ask_question(embeddings, text):
    """Ask a question and get the most similar QA pair."""
    t0 = time.time()
    # Escape single quotes in the query text
    escaped_text = text.replace("'", "''")
    
    try:
        results = embeddings.search(
            f"select text, answer, score from txtai where similar('{escaped_text}') limit 1"
        )
        
        if results:
            result = results[0]
            print(f"\nYour question: {text}")
            print(f"Most similar question: {result['text']}")
            print(f"Answer: {result['answer']}")
            print(f"Similarity score: {result['score']:.4f}")
            print(f"Time to search: {time.time() - t0:.2f}s")
        else:
            print("No matching answer found.")
    except Exception as e:
        print(f"\nError processing question: {str(e)}")
        print("Try rephrasing your question without special characters.")

def main():
    t0_total = time.time()
    # Load dataset and build index with caching
    dataset = load_qa_dataset(use_cache=True)
    embeddings = build_qa_index(dataset, use_cache=True)
    print(f"\nTotal initialization time: {time.time() - t0_total:.2f}s")
    
    # Test questions
    test_questions = [
        "What is the timezone of NYC?",
        "Things to do in New York",
        "What is the timezone of Florida?",
        "Who is Justin Bieber brother",  
        "What role did Natalie Portman have in Star Wars?"
    ]
    
    print("\nTesting pre-defined questions:")
    print("-" * 50)
    for question in test_questions:
        ask_question(embeddings, question)
        print()
    
    # Interactive mode
    print("\nEnter your own questions (type 'q' to quit):")
    print("-" * 50)
    while True:
        question = input("\nYour question (or 'q' to quit): ")
        if question.lower() == 'q':
            break
        ask_question(embeddings, question)

if __name__ == "__main__":
    main()
