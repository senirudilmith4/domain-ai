import re
import os
import hashlib
from pathlib import Path
from typing import List, Tuple
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter


SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_PATH = SCRIPT_DIR.parent / "data" / "docs"

EMBED_MODEL = "all-MiniLM-L6-v2"  # Upgraded from all-MiniLM-L6-v2 for better domain retrieval
_embed_model = None

# Per-doc-type chunk sizes: (chunk_size, chunk_overlap)
CHUNK_SIZES = {
    "module_descriptor": (600, 100),
    "policy":            (400, 80),
    "form":              (300, 50),
    "default":           (500, 100),
}


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def get_splitter(doc_type: str) -> SentenceSplitter:
    """Return a SentenceSplitter tuned for the given document type."""
    size, overlap = CHUNK_SIZES.get(doc_type, CHUNK_SIZES["default"])
    return SentenceSplitter(chunk_size=size, chunk_overlap=overlap)


def load_pdf(path: str) -> List[str]:
    """Load a PDF and return a list of page-level text strings."""
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    if not texts:
        raise ValueError(f"No text extracted from {path}. Please check the PDF file.")
    return texts


def split_by_sections(text: str) -> List[str]:
    """
    Split text on common structural markers found in university documents:
    - Numbered sections:  1.  2.  10.
    - Lettered points:    a.  b.
    - Roman numerals:     I.  IV.  XII.
    - ALL-CAPS headings:  POLICY  ASSESSMENT CRITERIA
    """
    pattern = r"\n\s*(?:\d+\.|[a-z]\.|[IVXLC]+\.|[A-Z]{2}[A-Z\s]*(?=\n|:))\s"
    sections = re.split(pattern, text)
    return [s.strip() for s in sections if s.strip()]


def chunk_texts(texts: List[str], doc_type: str = "default") -> List[str]:
    """
    Split and chunk a list of page texts.
    Uses section-aware splitting first, then SentenceSplitter for size control.
    doc_type controls chunk_size / overlap via CHUNK_SIZES.
    """
    splitter = get_splitter(doc_type)
    chunks = []
    for t in texts:
        sections = split_by_sections(t)
        for sec in sections:
            chunks.extend(splitter.split_text(sec))
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_embed_model()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    return embeddings.tolist()


def chunk_id(chunk: str, source: str) -> str:
    """
    Deterministic ID from content + source path.
    Allows ChromaDB upsert to overwrite stale chunks on re-ingestion
    instead of creating duplicates.
    """
    return hashlib.md5(f"{source}:{chunk}".encode()).hexdigest()


def sample_texts_for_metadata(texts: List[str]) -> str:
    """
    Sample start, middle, and end pages for more representative
    metadata detection — avoids missing content that starts after a
    long preamble (common in policy documents).
    """
    if not texts:
        return ""
    indices = sorted({0, len(texts) // 2, len(texts) - 1})
    return " ".join(texts[i] for i in indices)