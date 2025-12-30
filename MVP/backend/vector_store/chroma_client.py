from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from MVP.backend.utils.logger import get_logger

logger = get_logger(__name__)

# Determine the database path relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
CHROMA_PATH = SCRIPT_DIR.parent / "data" / "chroma_db"


class ChromaVectorStore:
    """Manages ChromaDB vector store for RAG system."""
    
    def __init__(
        self,
        persist_directory: Path = CHROMA_PATH,
        collection_name: str = "documents"
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
        """
        self.persist_directory = Path(persist_directory).resolve()
        self.collection_name = collection_name
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection: Optional[chromadb.Collection] = None
        
    @property
    def client(self) -> chromadb.ClientAPI:
        """Lazy load the ChromaDB client."""
        if self._client is None:
            self._client = self._initialize_client()
        return self._client
    
    @property
    def collection(self) -> chromadb.Collection:
        """Lazy load the collection."""
        if self._collection is None:
            self._collection = self._initialize_collection()
        return self._collection
    
    def _initialize_client(self) -> chromadb.ClientAPI:
        """Initialize ChromaDB client with persistence."""
        try:
            # Ensure directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Initializing ChromaDB at {self.persist_directory}")
            
            client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True  # Useful for development
                )
            )
            
            logger.info("ChromaDB client initialized successfully")
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
            raise
    
    def _initialize_collection(self) -> chromadb.Collection:
        """Initialize or get existing collection."""
        try:
            logger.info(f"Getting or creating collection: {self.collection_name}")
            
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.info(
                f"Collection '{self.collection_name}' ready. "
                f"Current document count: {collection.count()}"
            )
            
            return collection
            
        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}", exc_info=True)
            raise
    
    def add_documents(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            chunks: Text chunks to store
            embeddings: Embedding vectors for chunks
            metadatas: Optional metadata for each chunk (e.g., source file, chunk index)
            ids: Optional unique IDs for each chunk
        
        Raises:
            ValueError: If chunks and embeddings length mismatch
        """
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings provided")
            return
        
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )
        
        try:
            # Generate IDs if not provided
            if ids is None:
                # Use collection count as offset for unique IDs
                current_count = self.collection.count()
                ids = [f"chunk_{current_count + i}" for i in range(len(chunks))]
            
            # Generate default metadata if not provided
            if metadatas is None:
                metadatas = [{"chunk_index": i} for i in range(len(chunks))]
            
            logger.info(f"Adding {len(chunks)} documents to collection '{self.collection_name}'")
            
            # Add to ChromaDB
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(
                f"Successfully added {len(chunks)} documents. "
                f"Total in collection: {self.collection.count()}"
            )
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}", exc_info=True)
            raise
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: Query embedding vector (384-dim for all-MiniLM-L6-v2)
            n_results: Number of top results to return
            where: Optional metadata filter (e.g., {"source": "file1.pdf"})
            where_document: Optional document content filter
        
        Returns:
            Dictionary with 'documents', 'distances', 'metadatas', and 'ids'
        """
        try:
            logger.info(f"Querying collection for top {n_results} similar documents")
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            num_results = len(results['documents'][0]) if results['documents'] else 0
            logger.info(f"Query returned {num_results} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}", exc_info=True)
            return {}
    
    def delete_collection(self) -> None:
        """Delete the current collection and all its data."""
        try:
            logger.warning(f"Deleting collection: {self.collection_name}")
            self.client.delete_collection(name=self.collection_name)
            self._collection = None
            logger.info(f"Collection '{self.collection_name}' deleted successfully")
            
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}", exc_info=True)
            raise
    
    def reset_database(self) -> None:
        """Reset the entire ChromaDB database (deletes all collections)."""
        try:
            logger.warning("Resetting entire ChromaDB database")
            self.client.reset()
            self._collection = None
            logger.info("Database reset complete")
            
        except Exception as e:
            logger.error(f"Failed to reset database: {e}", exc_info=True)
            raise
    
    def update_document(
        self,
        chunk_id: str,
        new_text: Optional[str] = None,
        new_embedding: Optional[List[float]] = None,
        new_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update an existing document in the collection.
        
        Args:
            chunk_id: ID of the chunk to update
            new_text: New text content (optional)
            new_embedding: New embedding vector (optional)
            new_metadata: New metadata (optional)
        """
        try:
            logger.info(f"Updating document with ID: {chunk_id}")
            
            update_kwargs = {"ids": [chunk_id]}
            
            if new_text is not None:
                update_kwargs["documents"] = [new_text]
            if new_embedding is not None:
                update_kwargs["embeddings"] = [new_embedding]
            if new_metadata is not None:
                update_kwargs["metadatas"] = [new_metadata]
            
            self.collection.update(**update_kwargs)
            logger.info(f"Document {chunk_id} updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update document: {e}", exc_info=True)
            raise
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete specific documents from the collection.
        
        Args:
            ids: List of document IDs to delete
        """
        try:
            logger.info(f"Deleting {len(ids)} documents")
            self.collection.delete(ids=ids)
            logger.info(f"Successfully deleted {len(ids)} documents")
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}", exc_info=True)
            raise


# Convenience function for quick initialization
def get_vector_store(collection_name: str = "documents") -> ChromaVectorStore:
    """
    Get a ChromaVectorStore instance.
    
    Args:
        collection_name: Name of the collection to use
    
    Returns:
        ChromaVectorStore instance
    """
    return ChromaVectorStore(collection_name=collection_name)