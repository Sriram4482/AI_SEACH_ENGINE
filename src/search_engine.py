# src/search_engine.py
from typing import List, Dict, Tuple, Optional
import re
import numpy as np
import faiss

from .preprocess import load_documents
from .cache_manager import CacheManager
from .embedder import Embedder


def normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors so we can use inner product as cosine similarity.
    Works on 2D arrays (n, dim) or single vector (1, dim).
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    return vectors / norms


def _tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer: split on non-letters/digits, filter short tokens.
    """
    tokens = re.split(r"[^a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 2]


class SearchEngine:
    """
    Loads documents, generates/caches embeddings, builds FAISS index,
    and handles semantic search with ranking explanations.

    Enhancements:
      - best_passage preview (sentence with highest token overlap)
      - combined_score: semantic + overlap + length norm
      - optional Cross-Encoder reranking (if sentence_transformers.CrossEncoder is available)
      - keyword highlighting in preview
    """

    def __init__(self):
        self.cache = CacheManager()
        self.embedder = Embedder()
        self.docs: List[Dict] = []
        self.index = None
        self.dim = None

        # try to import CrossEncoder for optional reranking
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            # choose a small local cross-encoder model
            self.cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            self.cross = None

    # ---------- Index building ----------

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
                embeddings_list.append(emb.astype("float32"))
                self.cache.set(doc_id, emb, doc_hash)

        self.cache.save()

        all_embeddings = np.vstack(embeddings_list)
        all_embeddings = normalize(all_embeddings)

        self.dim = all_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)  # inner product = cosine on normalized vectors
        self.index.add(all_embeddings)
        print(f"✅ Built FAISS index with {len(self.docs)} documents, dim={self.dim}")

    # ---------- Explanation / Ranking helpers ----------

    def _explain_match(self, query: str, doc: Dict) -> Dict:
        """
        Heuristic explanation:
          - matched keywords
          - overlap ratio (matched / query tokens)
          - length normalization score (prefer concise docs slightly)
        """
        query_tokens = set(_tokenize(query))
        doc_tokens = set(_tokenize(doc["clean_text"]))

        common = query_tokens & doc_tokens
        overlap_ratio = 0.0
        if query_tokens:
            overlap_ratio = len(common) / len(query_tokens)

        # length normalization: prefer more concise docs (small boost)
        length = max(doc.get("length", 1), 1)
        import math

        length_norm = 1.0 / math.log(1.0 + length)

        return {
            "matched_keywords": sorted(list(common)),
            "overlap_ratio": overlap_ratio,
            "length_normalization_score": length_norm,
        }

    def best_passage(self, text: str, query: str, max_len: int = 250) -> str:
        """
        Return the best matching sentence for the query (highest token overlap).
        Falls back to the beginning of the document if no sentence matches.
        """
        query_tokens = set(_tokenize(query))
        # split into sentences reasonably
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        best = ""
        best_score = -1
        for sent in sentences:
            sent_tokens = set(_tokenize(sent))
            score = len(query_tokens & sent_tokens)
            if score > best_score:
                best_score = score
                best = sent

        if not best:
            # fallback: first max_len characters
            snippet = text.strip()[:max_len]
            return snippet + ("..." if len(text.strip()) > max_len else "")
        if len(best) > max_len:
            return best[:max_len] + "..."
        return best

    def highlight(self, snippet: str, keywords: List[str]) -> str:
        """
        Simple highlighting by wrapping matched keywords with ** markers.
        Keeps case-insensitive replacement by using regex word boundaries.
        """
        if not keywords:
            return snippet
        # sort by length desc to avoid partial overlaps (e.g., "learn" before "learned")
        keywords_sorted = sorted(set(keywords), key=lambda x: -len(x))
        out = snippet
        for kw in keywords_sorted:
            if not kw:
                continue
            # regex replace (case-insensitive, word boundaries)
            try:
                out = re.sub(rf"(?i)\b{re.escape(kw)}\b", lambda m: f"**{m.group(0)}**", out)
            except re.error:
                # fallback naive replace
                out = out.replace(kw, f"**{kw}**")
        return out

    # ---------- Search ----------

    def search(self, query: str, top_k: int = 5, rerank_top: Optional[int] = 10) -> List[Dict]:
        """
        1. Embed query
        2. Search FAISS index to get top `rerank_top` candidates
        3. Optionally rerank candidates with Cross-Encoder (if available)
        4. Compute explanations and combined score (semantic + heuristics)
        5. Return top_k results sorted by combined score
        """
        if self.index is None or not self.docs:
            raise RuntimeError("Index not built. Call build_index() first.")

        # embed and normalize query
        q_emb = self.embedder.encode_one(query)
        q_emb = np.asarray(q_emb, dtype="float32").reshape(1, -1)
        q_emb = normalize(q_emb)

        # decide how many candidates to fetch from FAISS for reranking
        fetch_k = min(max(rerank_top or top_k, top_k), len(self.docs))

        scores, indices = self.index.search(q_emb, fetch_k)
        scores = scores[0]
        indices = indices[0]

        # build initial candidate list
        candidates = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.docs):
                continue
            doc = self.docs[idx]
            candidates.append(
                {
                    "doc_index": idx,
                    "doc": doc,
                    "faiss_score": float(score),
                }
            )

        # Optional Cross-Encoder reranking for better precision (if available)
        if self.cross is not None and candidates:
            try:
                pairs = [(query, c["doc"]["clean_text"]) for c in candidates]
                cross_scores = self.cross.predict(pairs)  # higher is better
                for c, cs in zip(candidates, cross_scores):
                    c["cross_score"] = float(cs)
                # sort by cross score descending
                candidates.sort(key=lambda x: x.get("cross_score", x["faiss_score"]), reverse=True)
            except Exception:
                # if Cross-Encoder fails, keep FAISS order
                pass

        results = []
        for c in candidates[:fetch_k]:
            doc = c["doc"]
            faiss_score = c.get("faiss_score", 0.0)

            explanation = self._explain_match(query, doc)
            # best passage preview
            preview = self.best_passage(doc["clean_text"], query)

            # highlight matched keywords
            preview_highlighted = self.highlight(preview, explanation.get("matched_keywords", []))

            # combined score: semantic + small heuristic boosts (tunable)
            # prefer cross_score if available (already used for sorting), but still include faiss_score
            sem_score = c.get("cross_score", faiss_score)
            combined_score = float(sem_score) + 0.12 * float(explanation.get("overlap_ratio", 0.0)) + 0.06 * float(
                explanation.get("length_normalization_score", 0.0)
            )

            results.append(
                {
                    "doc_id": doc["doc_id"],
                    "score": float(sem_score),
                    "combined_score": float(combined_score),
                    "preview": preview_highlighted,
                    "explanation": explanation,
                    "metadata": {
                        "length": doc.get("length"),
                        # keep only filename for privacy/usability
                        "path": doc.get("path"),
                    },
                }
            )

        # final sort by combined_score and return top_k
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:top_k]
