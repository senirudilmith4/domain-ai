from typing import List, Optional
from sentence_transformers import SentenceTransformer
from MVP.backend.utils.logger import get_logger

logger = get_logger(__name__)

# Lazy loading pattern
_model: Optional[SentenceTransformer] = None

def get_embedding_model() -> SentenceTransformer:
    """Lazy load the embedding model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: all-MiniLM-L6-v2")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def embed_documents(
    chunks: List[str],
    batch_size: int = 32,
    show_progress: bool = True
) -> List[List[float]]:
    """
    Converts text chunks into embedding vectors.
    
    Args:
        chunks (List[str]): List of text chunks
        batch_size (int): Number of chunks to process at once
        show_progress (bool): Whether to show progress bar
    
    Returns:
        List[List[float]]: Embedding vectors (384-dimensional for all-MiniLM-L6-v2)
    """
    if not chunks:
        logger.warning("No chunks provided for embedding")
        return []
    
    # Filter out empty chunks
    valid_chunks = [chunk for chunk in chunks if chunk and chunk.strip()]
    
    if len(valid_chunks) != len(chunks):
        logger.warning(
            f"Filtered out {len(chunks) - len(valid_chunks)} empty chunks"
        )
    
    if not valid_chunks:
        logger.warning("No valid chunks after filtering")
        return []
    
    try:
        logger.info(f"Generating embeddings for {len(valid_chunks)} chunks")  # Log the number of valid chunks
        model = get_embedding_model()
        
        embeddings = model.encode( 
            valid_chunks,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        logger.info(
            f"Embedding generation completed. "
            f"Shape: {embeddings.shape}, Dtype: {embeddings.dtype}"
        )
        
        return embeddings.tolist()
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
        raise