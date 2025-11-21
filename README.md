# Multi-Document Embedding Search Engine

**Semantic search over local text documents** using sentence-transformers embeddings + FAISS, with caching, reranking, best-passage previews, and a Streamlit UI.

---

## Quick links (local files)

* Backend API: `/mnt/data/api.py`
* Search engine core: `/mnt/data/search_engine.py`
* Build index script: `/mnt/data/build_index.py`
* Preprocessing: `/mnt/data/preprocess.py`
* Embedder wrapper: `/mnt/data/embedder.py`
* Streamlit UI: `/mnt/data/ui.py`

> The paths above point to files in the project workspace. When viewed on your machine or CI, they will be available under the repository tree.

---

## Requirements

Install dependencies (recommended in a virtualenv):

```bash
python -m pip install -r requirements.txt
# or install minimal packages
pip install sentence-transformers faiss-cpu fastapi uvicorn streamlit requests
```

> On Windows, `faiss-cpu` might need a specific wheel or installation via conda. If `faiss-cpu` fails to install, you can run with a smaller dataset or use an alternative vector index.

---

## Project structure

```
./
├─ src/
│  ├─ api.py                # FastAPI app (POST /search)
│  ├─ search_engine.py      # Core search logic (FAISS, reranking, explanations)
│  ├─ build_index.py        # Script to (re)build FAISS index
│  ├─ preprocess.py        # Load & clean text files from data/docs
│  ├─ embedder.py          # Local sentence-transformers wrapper
│  └─ cache_manager.py     # Cache embeddings & metadata
├─ ui.py                   # Streamlit front-end
├─ data/docs/              # Put your .txt documents here
├─ data/faiss.index        # Generated FAISS index (after build)
├─ data/metadata.json      # Saved metadata for documents
└─ requirements.txt
```

---

## How to run (local)

1. **Prepare documents**

   * Put plain `.txt` files into `data/docs/`. Each file name (without extension) becomes `doc_id`.

2. **Build the index**

```bash
# Recommended: run as module so relative imports work
python -m src.build_index
# or
python src/build_index.py    # only if you adjusted sys.path or use absolute imports
```

This will:

* preprocess documents (`src/preprocess.py`),
* create embeddings (cached via `cache_manager`),
* build and save a FAISS index (fast to load on subsequent runs).

3. **Start the backend API**

```bash
uvicorn src.api:app --reload
```

* API base URL: `http://127.0.0.1:8000`
* Docs (interactive): `http://127.0.0.1:8000/docs`

4. **Start the UI (optional)**

```bash
streamlit run ui.py
```

The UI calls the backend `POST /search` endpoint and displays results with highlighted best passages and explanations.

---

## API

**POST /search**

Request body (JSON):

```json
{ "query": "your search text", "top_k": 5 }
```

Response (example):

```json
{
  "results": [
    {
      "doc_id": "doc_015",
      "score": 5.60,
      "combined_score": 5.73,
      "preview": "...best matching sentence with **highlights**...",
      "explanation": { "matched_keywords": [...], "overlap_ratio": 1.0, "length_normalization_score": 0.22 },
      "metadata": { "length": 95, "path": "data/docs/doc_015.txt" }
    }
  ]
}
```

---

## Design notes

* **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, CPU-friendly). The wrapper `src/embedder.py` exposes `encode_one` and `encode_batch`.
* **Index**: FAISS `IndexFlatIP` with normalized vectors (inner product used as cosine similarity).
* **Reranking**: optional Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is used if available — it yields higher precision at cost of a one-time model download.
* **Passages**: Documents are split into sentences; the UI shows the best matching sentence (highest token overlap) with matched keywords highlighted.
* **Caching**: Embeddings are cached per document (by SHA256 hash) to avoid re-embedding unchanged files.

---

## Tips & troubleshooting

* If you see TF oneDNN messages during model load, they are informational only.
* If Cross-Encoder downloads slowly, run without reranking (the system works fine with FAISS-only).
* On Windows if `faiss-cpu` fails to install, consider using `conda` or running on WSL.


## License

Documented created by SRIRAM CHODAVARAPU....



