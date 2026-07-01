import streamlit as st
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_dataset():
    print("Loading dataset...")
    return pd.read_pickle(
        "data/processed/papers.pkl"
    )

@st.cache_resource
def load_index():
    print("Loading index...")
    return faiss.read_index(
        "models/paper_index.faiss"
    )

@st.cache_resource
def load_model():
    print("Loading model...")
    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

df = load_dataset()
index = load_index()
model = load_model()

def search(query, k=5):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Normalize the query vector to unit length, same as the index vectors.
    # Without this, the inner product is not a true cosine similarity score.
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for idx, score in zip(
        indices[0],
        scores[0]
    ):
        # score is cosine similarity in [-1, 1].
        # Clamp to [0, 1] then multiply by 100 for a clean percentage.
        similarity = round(
            float(max(0.0, min(1.0, score))) * 100,
            1
        )

        paper = df.iloc[idx]

        results.append({
            "title": paper["title"],
            "authors": paper["authors"],
            "categories": paper["categories"],
            "abstract": paper["abstract"],
            "similarity": f"{similarity}%"
        })

    return results