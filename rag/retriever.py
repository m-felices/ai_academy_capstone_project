import streamlit as st
from pathlib import Path
from rag.loader import load_docs
from rag.vectorstore import build_vectorstore
from rag.keyword_search import keyword_search
from rag.text_splitter import split_into_chunks
from utils.logger import setup_logger

logger = setup_logger("retriever")

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

@st.cache_resource
def init_retrievers():
    """Create or load vectorstore + BM25 retriever"""
    docs = load_docs()
    chunks = split_into_chunks(docs)

    vectorstore = build_vectorstore(chunks, CHROMA_DIR)

    bm25 = keyword_search(chunks)
    return vectorstore, bm25, chunks


def retrieve_docs(query, k=3):
    """Retrieve top documents (hybrid Chroma + BM25)"""
    try:
        logger.info(f"Retrieving docs for query: {query}")
        vectorstore, bm25, _chunks = init_retrievers()

        # Vector retrieval
        vector_docs = vectorstore.search(query, search_type='similarity', k=k)
        #vector_docs = vectorstore.similarity_search(query, k=k)

        bm25_docs  = bm25.get_relevant_documents(query, k=k)

        logger.info(f"Vector search returned {len(vector_docs)} docs")
        logger.info(f"Keyword search returned {len(bm25_docs)} docs")

        # Combine and deduplicate
        combined = vector_docs + bm25_docs
        logger.info(f"Total unique docs: {len(combined)}")
        seen = set()
        unique = []
        for d in combined:
            text = d.page_content
            if text not in seen:
                seen.add(text)
                unique.append(d)

        return unique[:k]
    except Exception as e:
        raise RuntimeError(f"retrieve_docs failed: {e}") from e