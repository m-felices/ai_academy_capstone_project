import hashlib
from pathlib import Path
from typing import Dict
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from utils.config import EMBEDDING_MODEL
from utils.logger import setup_logger
logger = setup_logger("vectorstore")

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

def _generate_id(content: str, metadata: Dict) -> str:
    """Generate deterministic ID for deduplication."""

    # Include multiple metadata fields to ensure uniqueness
    unique_str = (
      f"{content}_"
      f"{metadata.get('source', '')}_"
      f"{metadata.get('page', 'no_page')}_"
      f"{metadata.get('chunk_index', 0)}_"
      f"{metadata.get('type', 'unknown')}"
    )
    return hashlib.md5(unique_str.encode()).hexdigest()

def build_vectorstore(chunks, persist_directory=CHROMA_DIR):
    try:
        logger.info("Building vectorstore")

        if not chunks:
            logger.error("No chunks provided")
            raise ValueError("No chunks")

        logger.info(f"Number of chunks: {len(chunks)}")

        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = Chroma(
            collection_name="pyansys",
            embedding_function=embeddings,
            persist_directory=str(persist_directory)
        )
        ids = [_generate_id(c.page_content, c.metadata) for c in chunks]

        vector_store.add_documents(documents=chunks, ids=ids)
        logger.info("Vectorstore built successfully")
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Evaluation failed: {e}")

def load_vectorstore(persist_directory=CHROMA_DIR):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings
    )
