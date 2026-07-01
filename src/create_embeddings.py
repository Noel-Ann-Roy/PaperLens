import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

# Load dataset
print("Loading papers...")

df = pd.read_csv(
    "data/processed/ai_papers.csv"
)

print("Total papers:", len(df))

texts = (
    df["title"].fillna("")
    + " "
    + df["abstract"].fillna("")
).tolist()

# Load model
print("Loading model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

BATCH_SIZE = 256

all_embeddings = []

print("Generating embeddings...")

for i in tqdm(
    range(0, len(texts), BATCH_SIZE)
):

    batch = texts[i:i+BATCH_SIZE]

    embeddings = model.encode(
        batch,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    all_embeddings.append(embeddings)

embeddings = np.vstack(all_embeddings)

# Normalize to unit length so that inner product == cosine similarity.
# Without this, the index uses L2 distance which has no meaningful
# percentage interpretation. After normalization, scores are in [0, 1]
# and can be cleanly multiplied by 100 to get a real similarity %.
faiss.normalize_L2(embeddings)

os.makedirs(
    "models",
    exist_ok=True
)

np.save(
    "models/paper_embeddings.npy",
    embeddings
)

print("\nEmbedding shape:", embeddings.shape)
print("Embeddings normalized: yes")
print("Saved successfully.")