# src/preprocess.py
import os
import re
import hashlib
from typing import List, Dict

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "docs")


def clean_text(text: str) -> str:
    """Lowercase, remove HTML tags, collapse spaces."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)       # remove HTML tags
    text = re.sub(r"\s+", " ", text).strip()   # normalize spaces
    return text


def compute_hash(text: str) -> str:
    """SHA256 hash of text, used for caching."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_documents() -> List[Dict]:
    """
    Load all .txt files from DOCS_DIR.
    Returns list of:
    {
        "doc_id": "doc_001",
        "path": "/full/path/to/doc_001.txt",
        "text": "...",
        "clean_text": "...",
        "length": int,
        "hash": "sha256..."
    }
    """
    docs = []
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)

    for filename in os.listdir(DOCS_DIR):
        if not filename.lower().endswith(".txt"):
            continue
        path = os.path.join(DOCS_DIR, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        cleaned = clean_text(raw)
        doc_hash = compute_hash(cleaned)
        doc_id = os.path.splitext(filename)[0]

        docs.append(
            {
                "doc_id": doc_id,
                "path": path,
                "text": raw,
                "clean_text": cleaned,
                "length": len(cleaned.split()),
                "hash": doc_hash,
            }
        )
    return docs
