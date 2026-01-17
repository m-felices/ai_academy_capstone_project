from rank_bm25 import BM25Okapi
from langchain_community.docstore.document import Document
from utils.logger import setup_logger

logger = setup_logger("keyword_search")

class BM25Retriever:

    def __init__(self, chunks):
        # Tokenize each chunk using split on whitespace, but normalize lowercase
        if not chunks:
            logger.warning("BM25 initialized with empty chunks list")
            self.texts = []
            self.metadata = []
            self.bm25 = None
            return

        self.texts = [c.page_content for c in chunks]
        logger.info(f"BM25 indexing {len(self.texts)} chunks")

        self.metadata = [c.metadata for c in chunks]
        self.tokenized_texts = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized_texts)
        logger.info("BM25 index built successfully")

    def get_relevant_documents(self, query, k=5):
        try:
            # Tokenize query same as corpus
            logger.info(f"BM25 retrieval for query: '{query}' (k={k})")

            if not self.bm25:
                logger.warning("BM25 retriever not initialized")
                return []

            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = scores.argsort()[-k:][::-1]
            results = []
            for idx in top_indices:
                results.append(Document(page_content=self.texts[idx], metadata=self.metadata[idx]))
            return results

        except Exception as e:
            raise RuntimeError(f"retrieve_docs failed: {e}")

def keyword_search(chunks):
    logger.info("Creating keyword (BM25) retriever")
    retriever = BM25Retriever(chunks)
    return retriever