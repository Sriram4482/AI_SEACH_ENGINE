# ui.py
import streamlit as st
import time
import os
import sys

# Ensure project root is importable (so `src` package imports work)
PROJECT_ROOT = os.path.dirname(__file__)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.search_engine import SearchEngine  # type: ignore


@st.cache_resource(show_spinner=True)
def load_engine() -> SearchEngine:
    """
    Initialize the SearchEngine and build the FAISS index once.
    Streamlit will cache this object across reruns.
    """
    engine = SearchEngine()
    start = time.time()
    engine.build_index()
    end = time.time()
    # note: calling st.* inside cached function can sometimes be surprising;
    # we return the engine and show the timing in the UI instead.
    engine._index_build_time = end - start  # attach for display
    return engine


def main():
    st.set_page_config(page_title="Semantic Search Engine", layout="wide")
    st.title("🔎 Multi-Document Embedding Search Engine")
    st.caption(
        "Semantic search over ~200 local documents with embeddings, caching, FAISS, and ranking explanations."
    )

    # Load engine (cached)
    engine = load_engine()

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write(
            """
            **How it works:**
            - Documents are preprocessed and embedded with MiniLM.
            - Embeddings are cached in a JSON file.
            - A FAISS index is built for fast similarity search.
            - Each result includes a simple ranking explanation.
            """
        )
        st.markdown("---")
        st.write("💾 **Docs folder:** `data/docs/`")
        st.write("⚙️ **Index:** FAISS IndexFlatIP (cosine similarity)")
        st.markdown("---")
        # show index build time if available
        build_time = getattr(engine, "_index_build_time", None)
        if build_time is not None:
            st.write(f"🕒 Index built in **{build_time:.2f} seconds**")

    # Main search controls
    query = st.text_input("Enter your search query:", value="machine learning applications")
    top_k = st.slider("Number of results (top_k):", min_value=1, max_value=20, value=5)

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a non-empty query.")
        else:
            with st.spinner("Searching..."):
                results = engine.search(query=query, top_k=top_k)

            if not results:
                st.info("No results found.")
            else:
                st.subheader("Search Results")
                for i, r in enumerate(results, start=1):
                    # show both semantic score and combined score (if present)
                    score = r.get("score", 0.0)
                    combined = r.get("combined_score", None)
                    if combined is not None:
                        header_text = f"### {i}. `{r['doc_id']}`  (score: `{score:.3f}`, combined: `{combined:.3f}`)"
                    else:
                        header_text = f"### {i}. `{r['doc_id']}`  (score: `{score:.3f}`)"
                    st.markdown(header_text)

                    # render the preview as markdown so **bold** highlights are visible
                    preview = r.get("preview", "")
                    st.markdown(preview, unsafe_allow_html=False)

                    exp = r.get("explanation", {})
                    meta = r.get("metadata", {})

                    with st.expander("View explanation & metadata"):
                        st.write("**Matched Keywords:**", ", ".join(exp.get("matched_keywords", [])) or "None")
                        st.write("**Overlap Ratio:**", f"{exp.get('overlap_ratio', 0):.3f}")
                        st.write(
                            "**Length Normalization Score:**",
                            f"{exp.get('length_normalization_score', 0):.3f}",
                        )
                        st.markdown("---")
                        st.write("**Document Length (tokens):**", meta.get("length"))
                        # show filename only for privacy/usability
                        file_path = meta.get("path") or ""
                        st.write("**File:**", os.path.basename(file_path))

                    st.markdown("---")


if __name__ == "__main__":
    main()
