import json
import os
import pandas as pd
from tqdm import tqdm

AI_CATEGORIES = {
    "cs.AI",
    "cs.LG",
    "cs.CL",
    "cs.CV",
    "stat.ML"
}

papers = []

print("Reading arXiv dataset...")

with open(
    "data/raw/arxiv-metadata-oai-snapshot.json",
    "r",
    encoding="utf-8"
) as f:

    for line in tqdm(f):

        paper = json.loads(line)

        categories = set(
            paper["categories"].split()
        )

        if categories & AI_CATEGORIES:

            papers.append({
                "title": paper["title"],
                "authors": paper["authors"],
                "abstract": paper["abstract"],
                "categories": paper["categories"],
                "year": paper["update_date"][:4]
            })

print("\nAI papers found:", len(papers))

os.makedirs(
    "data/processed",
    exist_ok=True
)

df = pd.DataFrame(papers)

df.to_csv(
    "data/processed/ai_papers.csv",
    index=False
)

print("\nCSV saved successfully!")
print("Location: data/processed/ai_papers.csv")