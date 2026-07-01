import json

with open(
    "data/raw/arxiv-metadata-oai-snapshot.json",
    "r",
    encoding="utf-8"
) as f:

    first_paper = json.loads(next(f))

print(first_paper.keys())
