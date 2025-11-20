# src/search_engine.py
from typing import List, Dict, Tuple
import numpy as np
import faiss

from .preprocess import load_documents
from .cache_manager import CacheManager
from .embedder import Embedder


def normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors so we can use inner product as cosine similarity.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    return vectors / norms


def _tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer: split on non-letters, filter short tokens.
    """
    import re

    tokens = re.split(r"[^a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 2]


class SearchEngine:
    """
    Loads documents, generates/caches embeddings, builds FAISS index,
    and handles semantic search with ranking explanation.
    """

    def __init__(self):
        self.cache = CacheManager()
        self.embedder = Embedder()
        self.docs: List[Dict] = []
        self.index = None
        self.dim = None

    def build_index(self) -> None:
        """
        1. Load & preprocess docs
        2. Use cache if hash matches; otherwise regenerate embeddings
        3. Build FAISS IndexFlatIP with normalized embeddings
        """
        self.docs = load_documents()
        if not self.docs:
            print("⚠ No documents found in data/docs. Add .txt files and rebuild.")
            return

        embeddings_list = []
        for doc in self.docs:
            doc_id = doc["doc_id"]
            doc_hash = doc["hash"]

            if self.cache.is_valid(doc_id, doc_hash):
                entry = self.cache.get(doc_id)
                emb = np.array(entry["embedding"], dtype="float32")
                embeddings_list.append(emb)
            else:
                emb = self.embedder.encode_one(doc["clean_text"])
                embeddings_list.append(emb)
                self.cache.set(doc_id, emb, doc_hash)

        self.cache.save()

        all_embeddings = np.vstack(embeddings_list)
        all_embeddings = normalize(all_embeddings)

        self.dim = all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)  # inner product
        self.index.add(all_embeddings)
        print(f"✅ Built FAISS index with {len(self.docs)} documents, dim={self.dim}")

    # ---------- Explanation / Ranking ----------

    def _explain_match(
        self, query: str, doc: Dict
    ) -> Dict:
        """
        Simple heuristic explanation:
        - keyword overlap
        - overlap ratio
        - length normalization score
        """
        query_tokens = set(_tokenize(query))
        doc_tokens = set(_tokenize(doc["clean_text"]))

        common = query_tokens & doc_tokens
        overlap_ratio = 0.0
        if query_tokens:
            overlap_ratio = len(common) / len(query_tokens)

        # length normalization: prefer more concise docs
        length = max(doc["length"], 1)
        import math

        length_norm = 1.0 / math.log(1.0 + length)

        return {
            "matched_keywords": sorted(list(common)),
            "overlap_ratio": overlap_ratio,
            "length_normalization_score": length_norm,
        }

    # ---------- Search ----------

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        1. Embed query
        2. Search FAISS index
        3. Attach explanations
        """
        if self.index is None or not self.docs:
            raise RuntimeError("Index not built. Call build_index() first.")

        q_emb = self.embedder.encode_one(query)
        q_emb = q_emb.reshape(1, -1)
        q_emb = normalize(q_emb)

        scores, indices = self.index.search(q_emb, top_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.docs):
                continue
            doc = self.docs[idx]
            explanation = self._explain_match(query, doc)

            preview = doc["clean_text"][:200] + ("..." if len(doc["clean_text"]) > 200 else "")

            results.append(
                {
                    "doc_id": doc["doc_id"],
                    "score": float(score),
                    "preview": preview,
                    "explanation": explanation,
                    "metadata": {
                        "length": doc["length"],
                        "path": doc["path"],
                    },
                }
            )
        return results
