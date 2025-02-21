from txtai import Embeddings

def main():
    # Create sample data
    data = [
        "US tops 5 million confirmed virus cases",
        "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
        "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
        "The National Park Service warns against sacrificing slower friends in a bear attack",
        "Maine man wins $1M from $25 lottery ticket",
        "Make huge profits without work, earn up to $100,000 a day"
    ]

    # Create embeddings model
    print("Initializing embeddings model...")
    embeddings = Embeddings(path="sentence-transformers/nli-mpnet-base-v2")

    # Index the data
    print("\nIndexing data...")
    embeddings.index(data)

    # Test queries
    queries = [
        "feel good story",
        "climate change",
        "public health story",
        "war",
        "wildlife",
        "asia",
        "lucky",
        "dishonest junk"
    ]

    # Run searches
    print("\nTesting semantic search:")
    print("%-20s %s" % ("Query", "Best Match"))
    print("-" * 50)

    for query in queries:
        # Get best match
        uid = embeddings.search(query, 1)[0][0]
        print("%-20s %s" % (query, data[uid]))

    # Test custom queries
    while True:
        query = input("\nEnter your search query (or 'q' to quit): ")
        if query.lower() == 'q':
            break

        # Get top 3 matches with scores
        results = embeddings.search(query, 3)
        print("\nTop 3 matches:")
        for uid, score in results:
            print(f"Score: {score:.4f} | Text: {data[uid]}")

if __name__ == "__main__":
    main()
