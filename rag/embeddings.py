from langchain_openai import OpenAIEmbeddings
from utils.config import EMBEDDING_MODEL

def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )