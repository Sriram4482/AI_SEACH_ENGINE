# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional

from .search_engine import SearchEngine

app = FastAPI(
    title="Multi-document Embedding Search Engine",
    description="Semantic search over local text documents with caching, FAISS, reranking & detailed explanations.",
    version="2.0.0",
)

# Initialize engine once
engine = SearchEngine()
engine.build_index()


# ---------- Request / Response Models ----------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    doc_id: str
    score: float
    combined_score: float
    preview: str
    explanation: Dict
    metadata: Dict


class SearchResponse(BaseModel):
    results: List[SearchResult]


# ---------- Root Endpoint ----------

@app.get("/", include_in_schema=False)
def root():
    return {
        "status": "ok",
        "message": "🚀 Backend running. Visit /docs for API documentation."
    }


# ---------- Search Endpoint ----------

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """
    Perform semantic search using embeddings + FAISS + reranking + passage extraction.
    """
    results = engine.search(req.query, req.top_k)
    return {"results": results}
