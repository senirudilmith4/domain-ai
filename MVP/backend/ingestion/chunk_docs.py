

from typing import List
from MVP.backend.utils.logger import get_logger

logger = get_logger(__name__)

def chunk_documents(
    documents: List[str],
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    '''
    This function takes a list of documents, slices each into smaller overlapping chunks of text, and returns all those chunks as a list.
    '''

    if overlap >= chunk_size:
        logger.error(
            f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )
        raise ValueError("Overlap must be less than chunk_size")

    logger.info(
        f"Starting document chunking with size={chunk_size}, overlap={overlap}"
    )

    chunks = []

    for doc_index, doc in enumerate(documents):

        if not doc or not doc.strip():
            logger.warning(
                f"Skipping empty document at index {doc_index}"
            )
            continue

        doc = doc.strip()
        doc_length = len(doc)
        start = 0
        doc_chunks = 0

        while start < doc_length:
            end = min(start + chunk_size, doc_length)
            chunk = doc[start:end].strip()

            if chunk:
                chunks.append(chunk)
                doc_chunks += 1

            start += chunk_size - overlap

        logger.info(
            f"Document {doc_index} ({doc_length} chars) chunked into {doc_chunks} chunks"
        )

    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks
