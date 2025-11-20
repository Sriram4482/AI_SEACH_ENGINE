# src/download_dataset.py
import os
from sklearn.datasets import fetch_20newsgroups

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "data", "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

print("Downloading dataset... (using fallback method)")

dataset = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes"),
    download_if_missing=True,
)
# <-- THIS FALLBACK WORKS ON SLOW NETWORKS.

max_docs = 200
count = 0

for i, text in enumerate(dataset.data[:max_docs]):
    filename = os.path.join(DOCS_DIR, f"doc_{i:03d}.txt")
    with open(filename, "w", encoding="utf-8", errors="ignore") as f:
        f.write(text)
    count += 1

print(f"Saved {count} documents to {DOCS_DIR}")
