# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from .search_engine import SearchEngine

app = FastAPI(
    title="Multi-document Embedding Search Engine",
    description="Semantic search over local text documents with caching & ranking explanation.",
    version="1.0.0",
)

# Global engine instance
engine = SearchEngine()
engine.build_index()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    doc_id: str
    score: float
    preview: str
    explanation: dict
    metadata: dict


class SearchResponse(BaseModel):
    results: List[SearchResult]


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    Semantic search endpoint.
    Input: { "query": "...", "top_k": 5 }
    Output: top_k documents with scores & explanation.
    """
    results = engine.search(req.query, req.top_k)
    return {"results": results}
