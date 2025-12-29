from pathlib import Path
import pdfplumber
from docx import Document
import os # Import os for potential error check
from MVP.backend.utils.logger import get_logger

logger = get_logger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_PATH = SCRIPT_DIR.parent / "data" / "docs"  # Create a Path object for the documents directory

def load_documents():
    logger.info(f"Loading documents from: {DOCS_PATH}")
    documents = []

    if not DOCS_PATH.exists():
        print(f"Error: Document path not found: {DOCS_PATH}")
        return documents

    for file in DOCS_PATH.iterdir():
        
        # Skip directories
        if file.is_dir():
            continue
            
        # --- Handle .txt files ---
        if file.suffix == ".txt":
            try:
                documents.append(file.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"Warning: Could not read TXT file {file.name}. Error: {e}")

        # --- Handle .pdf files ---
        elif file.suffix == ".pdf":
            try:
                text = ""
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                documents.append(text)
            except Exception as e:
                print(f"Warning: Could not process PDF file {file.name}. Error: {e}")

        # --- Handle .docx files ---
        elif file.suffix == ".docx":
            try:
                doc = Document(file)
                text = "\n".join(p.text for p in doc.paragraphs)
                documents.append(text)
            except Exception as e:
                print(f"Warning: Could not process DOCX file {file.name}. Error: {e}")
                
        # --- Handle unlisted file types (Optional) ---
        else:
            print(f"Skipping file with unhandled extension: {file.name}")


    return documents