import pandas as pd

print("Loading CSV...")

df = pd.read_csv(
    "data/processed/ai_papers.csv"
)

print("Saving pickle file...")

df.to_pickle(
    "data/processed/papers.pkl"
)

print("Done.")