from typing import List
from langchain_community.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import setup_logger

logger = setup_logger("text_splitter")

def split_into_chunks(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 50) -> List[Document]:
    """Split Documents into smaller chunks for vector indexing."""
    logger.info("Starting text splitting")
    logger.info(
        f"Parameters → chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
    )

    if not documents:
        logger.warning("No documents provided to text_splitter")
        return []


    splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
    )
    try:
        all_chunks = []
        for doc in documents:
            if not isinstance(doc, Document):
                raise TypeError(f"All items must be Document objects, got {type(doc)}")
            new_chunks = splitter.split_documents([doc])
            all_chunks.extend(new_chunks)

        return all_chunks
    except Exception as e:
        logger.exception("Text splitting failed")
        raise RuntimeError(f"split_into_chunks failed: {e}") from e

