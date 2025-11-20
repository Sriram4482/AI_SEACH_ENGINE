# src/cache_manager.py
import json
import os
from typing import Dict, Optional
from datetime import datetime

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
CACHE_FILE = os.path.join(CACHE_DIR, "embeddings_cache.json")


class CacheManager:
    """
    Simple JSON-based cache:
    {
      "doc_id": {
          "hash": "...",
          "embedding": [...],
          "updated_at": "2025-11-20T18:00:00"
      }
    }
    """

    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = cache_file
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        self._cache: Dict[str, Dict] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                try:
                    self._cache = json.load(f)
                except json.JSONDecodeError:
                    self._cache = {}
        else:
            self._cache = {}

    def save(self) -> None:
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)

    def get(self, doc_id: str) -> Optional[Dict]:
        return self._cache.get(doc_id)

    def is_valid(self, doc_id: str, current_hash: str) -> bool:
        entry = self._cache.get(doc_id)
        if not entry:
            return False
        return entry.get("hash") == current_hash

    def set(self, doc_id: str, embedding, doc_hash: str) -> None:
        self._cache[doc_id] = {
            "hash": doc_hash,
            "embedding": list(map(float, embedding)),
            "updated_at": datetime.utcnow().isoformat(),
        }

    def all_items(self) -> Dict[str, Dict]:
        return self._cache
