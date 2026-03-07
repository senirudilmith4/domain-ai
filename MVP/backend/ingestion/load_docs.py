from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
import os # Import os for potential error check
#from backend.utils.logger import get_logger
from typing import List
from pathlib import Path 


# logger = get_logger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_PATH = SCRIPT_DIR.parent / "data" / "docs"  # Create a Path object for the documents directory

# EMBED_MODEL = "all-MiniLM-L6-v2"  # Specify the embedding model to use
# embed_model = SentenceTransformer(EMBED_MODEL)  # Load the embedding model
splitter = SentenceSplitter(chunk_size=500, chunk_overlap=100)  
EMBED_MODEL = "all-MiniLM-L6-v2"
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model

def load_and_chunk_pdf(path: str):
    docs= PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]  # Ensure we only process nodes that have text content
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

     

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()