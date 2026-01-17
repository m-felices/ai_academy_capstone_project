from pathlib import Path

EMBEDDING_MODEL = "text-embedding-3-large"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PDF_DIR = DATA_DIR / "pdfs"
AUDIO_DIR = DATA_DIR / "audio"
AUDIO_CHUNK_SECONDS = 600  # 10 min chunks