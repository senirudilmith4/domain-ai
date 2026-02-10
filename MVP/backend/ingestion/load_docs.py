from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
import os # Import os for potential error check
from MVP.backend.utils.logger import get_logger
from sentence_transformers import SentenceTransformer
from typing import List


logger = get_logger(__name__)
# SCRIPT_DIR = Path(__file__).resolve().parent
# DOCS_PATH = SCRIPT_DIR.parent / "data" / "docs"  # Create a Path object for the documents directory

EMBED_MODEL = "all-MiniLM-L6-v2"  # Specify the embedding model to use
embed_model = SentenceTransformer(EMBED_MODEL)  # Load the embedding model
splitter = SentenceSplitter(chunk_size=500, chunk_overlap=100)  


def load_and_chunk_pdf(path: str):
    docs= PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]  # Ensure we only process nodes that have text content
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

    # if not DOCS_PATH.exists():
    #     print(f"Error: Document path not found: {DOCS_PATH}")
    #     return documents

    # for file in DOCS_PATH.iterdir():
        
    #     # Skip directories
    #     if file.is_dir():
    #         continue
            
    #     # --- Handle .txt files ---
    #     if file.suffix == ".txt":
    #         try:
    #             documents.append(file.read_text(encoding="utf-8"))
    #         except Exception as e:
    #             print(f"Warning: Could not read TXT file {file.name}. Error: {e}")

    #     # --- Handle .pdf files ---
    #     elif file.suffix == ".pdf":
    #         try:
    #             text = ""
    #             with pdfplumber.open(file) as pdf:
    #                 for page in pdf.pages:
    #                     text += page.extract_text() or ""
    #             documents.append(text)
    #         except Exception as e:
    #             print(f"Warning: Could not process PDF file {file.name}. Error: {e}")

    #     # --- Handle .docx files ---
    #     elif file.suffix == ".docx":
    #         try:
    #             doc = Document(file)
    #             text = "\n".join(p.text for p in doc.paragraphs)
    #             documents.append(text)
    #         except Exception as e:
    #             print(f"Warning: Could not process DOCX file {file.name}. Error: {e}")
                
    #     # --- Handle unlisted file types (Optional) ---
    #     else:
    #         print(f"Skipping file with unhandled extension: {file.name}")


    # return documents

def embed_texts(texts: list[str])-> list[list[float]]:
    embeddings = embed_model.encode(texts, show_progress_bar=False)  # Encode the texts into embeddings
    return embeddings.tolist()  # Convert numpy arrays to lists