import io
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from openai import OpenAI
from pydub import AudioSegment
from utils.config import PDF_DIR, AUDIO_CHUNK_SECONDS, AUDIO_DIR
from utils.logger import setup_logger

logger = setup_logger("loader")

client = OpenAI()


# -----------------------------
# PDF Loader
# -----------------------------
def _load_pdf_docs(pdf_dir: Path = PDF_DIR) -> List[Document]:
    """Load all PDFs from a directory and return a list of Documents."""
    logger.info(f"Loading PDFs from {PDF_DIR}")
    docs = []
    for pdf_file in pdf_dir.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            pdf_docs = loader.load()
            for i, doc in enumerate(pdf_docs):
                # Add metadata for filename and page
                doc.metadata["source"] = pdf_file.name
                doc.metadata["page"] = i + 1
            docs.extend(pdf_docs)
        except Exception as e:
            logger.exception(f"Failed to load {pdf_file.name}")
            raise RuntimeError(f"load_pdf_docs failed: {e}")
    return docs


# -----------------------------
# Audio Utilities
# -----------------------------
def _split_audio(file_path: Path, chunk_seconds: int = AUDIO_CHUNK_SECONDS) -> List[AudioSegment]:
    """Split audio into fixed-size chunks (milliseconds)."""
    try:
        audio = AudioSegment.from_file(str(file_path))
        duration_ms = len(audio)
        chunks = []
        for i in range(0, duration_ms, chunk_seconds * 1000):
            end = min(i + chunk_seconds * 1000, duration_ms)
            chunk = audio[i:end]
            if len(chunk) > 0:
                chunks.append(chunk)
        return chunks
    except Exception as e:
        logger.exception(f"Failed to aplit audio {e}")
        raise RuntimeError(f"split_audio failed: {e}")


# -----------------------------
# Audio Transcription
# -----------------------------
def _transcribe_long_audio(file_path: Path) -> List[Document]:
    """Transcribe a long audio file into chunked Documents."""
    try:
        audio_chunks = _split_audio(file_path)
        documents = []

        for i, chunk in enumerate(audio_chunks):
            buffer = io.BytesIO()
            # Export as WAV for best transcription quality
            chunk.export(
                buffer,
                format="wav",
                parameters=["-ac", "1", "-ar", "16000"]  # mono, 16kHz
            )
            buffer.seek(0)

            transcription = client.audio.transcriptions.create(
                file=("chunk.wav", buffer, "audio/wav"),
                model="gpt-4o-transcribe"
            )

            # Wrap each transcription chunk as Document
            documents.append(
                Document(
                    page_content=transcription.text,
                    metadata={
                        "source": file_path.name,
                        "chunk_index": i,
                        "start_sec": i * AUDIO_CHUNK_SECONDS,
                        "end_sec": (i + 1) * AUDIO_CHUNK_SECONDS
                    }
                )
            )

        return documents
    except Exception as e:
        logger.exception(f"Failed to transcribe long audio {e}")
        raise RuntimeError(f"load_audio_docs failed: {e}")


def _load_audio_docs(audio_dir: Path = AUDIO_DIR) -> List[Document]:
    """Load all audio files from a directory and transcribe them into Documents."""
    logger.info(f"Loading Audio files from {AUDIO_DIR}")

    all_docs = []
    for audio_file in audio_dir.glob("*.*"):
        try:
            logger.info(f"Loading {audio_file.name}")
            docs = _transcribe_long_audio(audio_file)
            all_docs.extend(docs)
        except Exception as e:
            logger.exception(f"Failed to load {audio_file.name}")
            raise RuntimeError(f"load_audio_docs failed: {e}")
    return all_docs


# ---------------------
# General loader
# ---------------------
def load_docs():
    """
    Load all document types (PDFs + audio files)
    """
    docs = [*_load_pdf_docs(), *_load_audio_docs()]
    return docs

