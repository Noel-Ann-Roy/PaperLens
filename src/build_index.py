

import numpy as np
import faiss
import os

print("Loading embeddings...")

embeddings = np.load(
    "models/paper_embeddings.npy"
)

print("Embedding shape:", embeddings.shape)

dimension = embeddings.shape[1]

# IndexFlatIP = inner product index.
# On L2-normalized vectors, inner product equals cosine similarity,
# which is bounded in [0, 1] for non-negative similarity.
# This replaces IndexFlatL2 whose distances were unbounded and
# made the "100 - distance" similarity formula meaningless.
index = faiss.IndexFlatIP(dimension)

index.add(
    embeddings.astype("float32")
)

os.makedirs(
    "models",
    exist_ok=True
)

faiss.write_index(
    index,
    "models/paper_index.faiss"
)

print("Index type: IndexFlatIP (cosine similarity)")
print("Index created successfully.")
print("Total vectors:", index.ntotal)