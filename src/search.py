import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv(
    "data/processed/ai_papers.csv"
)

# Load FAISS index
index = faiss.read_index(
    "models/paper_index.faiss"
)

# Load embedding model
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

while True:

    query = input(
        "\nEnter your research topic (or type exit): "
    )

    if query.lower() == "exit":
        break

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Normalize so inner product == cosine similarity (matches the index)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        5
    )

    print("\nRecommended Papers:\n")

    for rank, (idx, score) in enumerate(
        zip(indices[0], scores[0]), 1
    ):
        similarity = round(float(score) * 100, 1)
        paper = df.iloc[idx]

        print(f"{rank}. {paper['title']}")
        print(f"   Authors:    {paper['authors']}")
        print(f"   Category:   {paper['categories']}")
        print(f"   Similarity: {similarity}%")
        print()