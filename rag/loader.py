import io
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pydub import AudioSegment
from utils.config import PDF_DIR, AUDIO_CHUNK_SECONDS, AUDIO_DIR
from utils.logger import setup_logger
from llm.client import client

logger = setup_logger("loader")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
# -----------------------------
# PDF Loader
# -----------------------------
def _load_pdf_docs(pdf_dir: Path = PDF_DIR) -> List[Document]:
    """Load all PDFs from a directory and return a list of Documents."""
    logger.info(f"Loading PDFs from {pdf_dir}")
    docs: List[Document] = []

    if not pdf_dir.exists():
        logger.warning(f"PDF directory does not exist: {pdf_dir}")
        return docs

    for pdf_file in pdf_dir.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            pdf_docs = loader.load()

            for i, doc in enumerate(pdf_docs):
                doc.metadata["source"] = pdf_file.name
                doc.metadata["page"] = i + 1

            docs.extend(pdf_docs)

        except Exception as e:
            logger.exception(f"Failed to load PDF {pdf_file.name}: {e}")

    logger.info(f"Loaded {len(docs)} PDF documents")
    return docs


# -----------------------------
# Audio Utilities
# -----------------------------
def _split_audio(
    file_path: Path,
    chunk_seconds: int = AUDIO_CHUNK_SECONDS
) -> List[AudioSegment]:
    """Split audio into fixed-size chunks."""
    audio = AudioSegment.from_file(str(file_path))
    duration_ms = len(audio)
    chunks = []

    for i in range(0, duration_ms, chunk_seconds * 1000):
        end = min(i + chunk_seconds * 1000, duration_ms)
        chunk = audio[i:end]
        if len(chunk) > 0:
            chunks.append(chunk)

    return chunks


# -----------------------------
# Audio Transcription
# -----------------------------
def _transcribe_long_audio(file_path: Path) -> List[Document]:
    """Transcribe a long audio file into chunked Documents."""
    documents: List[Document] = []

    audio_chunks = _split_audio(file_path)

    for i, chunk in enumerate(audio_chunks):
        try:
            buffer = io.BytesIO()
            chunk.export(
                buffer,
                format="wav",
                parameters=["-ac", "1", "-ar", "16000"]
            )
            buffer.seek(0)

            transcription = client.audio.transcriptions.create(
                file=("chunk.wav", buffer, "audio/wav"),
                model="gpt-4o-transcribe"
            )

            documents.append(
                Document(
                    page_content=transcription.text,
                    metadata={
                        "source": file_path.name,
                        "type": "audio",
                        "chunk_index": i,
                        "start_sec": i * AUDIO_CHUNK_SECONDS,
                        "end_sec": (i + 1) * AUDIO_CHUNK_SECONDS,
                    },
                )
            )

        except Exception as e:
            logger.exception(
                f"Failed to transcribe chunk {i} of {file_path.name}: {e}"
            )

    return documents


def _load_audio_docs(audio_dir: Path = AUDIO_DIR) -> List[Document]:
    """Load all audio files from a directory and transcribe them."""
    logger.info(f"Loading Audio files from {audio_dir}")
    all_docs: List[Document] = []

    if not audio_dir.exists():
        logger.warning(f"Audio directory does not exist: {audio_dir}")
        return all_docs

    audio_files = [
        f for f in audio_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]

    if not audio_files:
        logger.info("No audio files found, skipping audio ingestion")
        return all_docs

    for audio_file in audio_files:
        logger.info(f"Processing audio file: {audio_file.name}")

        try:
            docs = _transcribe_long_audio(audio_file)
            all_docs.extend(docs)
        except Exception as e:
            logger.exception(
                f"Failed to process audio file {audio_file.name}: {e}"
            )

    logger.info(f"Loaded {len(all_docs)} audio documents")
    return all_docs


# ---------------------
# General loader
# ---------------------
def load_docs() -> List[Document]:
    """
    Load all document types (PDFs + audio files)
    """
    logger.info("Starting document loading pipeline")

    pdf_docs = _load_pdf_docs()
    audio_docs = _load_audio_docs()

    docs = [*pdf_docs, *audio_docs]

    logger.info(f"Total documents loaded: {len(docs)}")
    return docs

