from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
# from MVP.backend.utils.logger import get_logger

# logger = get_logger(__name__)

# Determine the database path relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
CHROMA_PATH = "data/chroma_db"




class ChromaVectorStore:
    """Manages ChromaDB vector store for RAG system."""

    def __init__(self, collection="docs", persist_directory=CHROMA_PATH):
        # Initialize Chroma client
        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            )
        )

        self.collection_name = collection

        # Create or get existing collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert(self,ids, chunks, embeddings, metadatas):      # Take text chunks from a document and safely store or update them in ChromaDB
            self.collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )


    def similarity_search(self, query_embedding, top_k=5, where=None):   # Perform a similarity search in ChromaDB using the query embedding and return the most relevant docs
        results = self.collection.query(
            query_embeddings=[query_embedding],   # Embedding vector for the search query
            n_results=top_k,    # Return the top K most similar documents based on cosine similarity
            where=where         # Optional metadata filter to narrow down search results (e.g., {"source": "document1.pdf"})
        )

        contexts = []
        sources = set()

        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        dists = results.get("distances", [])

        if docs and metas:
            for doc, meta in zip(docs[0], metas[0]):
                if doc:
                    contexts.append(doc)
                    if meta and "source" in meta:
                        sources.add(meta["source"])

        return {
            "contexts": contexts,
            "sources": list(sources),
            "scores": dists[0] if dists else []
        }

        
        
    # @property
    # def client(self) -> chromadb.ClientAPI:
    #     """Lazy load the ChromaDB client."""
    #     if self._client is None:
    #         self._client = self._initialize_client()
    #     return self._client
    
    # @property
    # def collection(self) -> chromadb.Collection:
    #     """Lazy load the collection."""
    #     if self._collection is None:
    #         self._collection = self._initialize_collection()
    #     return self._collection
    
    # def _initialize_client(self) -> chromadb.ClientAPI:
    #     """Initialize ChromaDB client with persistence."""
    #     try:
    #         # Ensure directory exists
    #         self.persist_directory.mkdir(parents=True, exist_ok=True)
            
    #         logger.info(f"Initializing ChromaDB at {self.persist_directory}")
            
    #         client = chromadb.PersistentClient(
    #             path=str(self.persist_directory),
    #             settings=Settings(
    #                 anonymized_telemetry=False,
    #                 allow_reset=True  # Useful for development
    #             )
    #         )
            
    #         logger.info("ChromaDB client initialized successfully")
    #         return client
            
    #     except Exception as e:
    #         logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
    #         raise
    
    # def _initialize_collection(self) -> chromadb.Collection:
    #     """Initialize or get existing collection."""
    #     try:
    #         logger.info(f"Getting or creating collection: {self.collection_name}")
            
    #         collection = self.client.get_or_create_collection(
    #             name=self.collection_name,
    #             metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    #         )
            
    #         logger.info(
    #             f"Collection '{self.collection_name}' ready. "
    #             f"Current document count: {collection.count()}"
    #         )
            
    #         return collection
            
    #     except Exception as e:
    #         logger.error(f"Failed to get/create collection: {e}", exc_info=True)
    #         raise
    
   
    
#     def get_collection_stats(self) -> Dict[str, Any]:
#         """Get statistics about the current collection."""
#         try:
#             count = self.collection.count()
#             return {
#                 "collection_name": self.collection_name,
#                 "total_documents": count,
#                 "persist_directory": str(self.persist_directory)
#             }
#         except Exception as e:
#             logger.error(f"Failed to get collection stats: {e}", exc_info=True)
#             return {}
    
#     def delete_collection(self) -> None:
#         """Delete the current collection and all its data."""
#         try:
#             logger.warning(f"Deleting collection: {self.collection_name}")
#             self.client.delete_collection(name=self.collection_name)
#             self._collection = None
#             logger.info(f"Collection '{self.collection_name}' deleted successfully")
            
#         except Exception as e:
#             logger.error(f"Failed to delete collection: {e}", exc_info=True)
#             raise
    
#     def reset_database(self) -> None:
#         """Reset the entire ChromaDB database (deletes all collections)."""
#         try:
#             logger.warning("Resetting entire ChromaDB database")
#             self.client.reset()
#             self._collection = None
#             logger.info("Database reset complete")
            
#         except Exception as e:
#             logger.error(f"Failed to reset database: {e}", exc_info=True)
#             raise
    
#     def update_document(
#         self,
#         chunk_id: str,
#         new_text: Optional[str] = None,
#         new_embedding: Optional[List[float]] = None,
#         new_metadata: Optional[Dict[str, Any]] = None
#     ) -> None:
#         """
#         Update an existing document in the collection.
        
#         Args:
#             chunk_id: ID of the chunk to update
#             new_text: New text content (optional)
#             new_embedding: New embedding vector (optional)
#             new_metadata: New metadata (optional)
#         """
#         try:
#             logger.info(f"Updating document with ID: {chunk_id}")
            
#             update_kwargs = {"ids": [chunk_id]}
            
#             if new_text is not None:
#                 update_kwargs["documents"] = [new_text]
#             if new_embedding is not None:
#                 update_kwargs["embeddings"] = [new_embedding]
#             if new_metadata is not None:
#                 update_kwargs["metadatas"] = [new_metadata]
            
#             self.collection.update(**update_kwargs)
#             logger.info(f"Document {chunk_id} updated successfully")
            
#         except Exception as e:
#             logger.error(f"Failed to update document: {e}", exc_info=True)
#             raise
    
#     def delete_documents(self, ids: List[str]) -> None:
#         """
#         Delete specific documents from the collection.
        
#         Args:
#             ids: List of document IDs to delete
#         """
#         try:
#             logger.info(f"Deleting {len(ids)} documents")
#             self.collection.delete(ids=ids)
#             logger.info(f"Successfully deleted {len(ids)} documents")
            
#         except Exception as e:
#             logger.error(f"Failed to delete documents: {e}", exc_info=True)
#             raise


# # Convenience function for quick initialization
# def get_vector_store(collection_name: str = "documents") -> ChromaVectorStore:
#     """
#     Get a ChromaVectorStore instance.
    
#     Args:
#         collection_name: Name of the collection to use
    
#     Returns:
#         ChromaVectorStore instance
#     """
#     return ChromaVectorStore(collection_name=collection_name)