# src/embedder.py
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wrapper around sentence-transformers/all-MiniLM-L6-v2
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into a 2D numpy array of shape (n, dim).
        """
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # Ensure float32 for FAISS
        return embeddings.astype("float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
