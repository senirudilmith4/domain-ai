from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
# from MVP.backend.utils.logger import get_logger


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
                anonymized_telemetry=False,
                is_persistent=True 
            )
        )

        self.collection_name = collection

        # Create or get existing collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print("Existing collections:", self.client.list_collections())

    def upsert(self,ids, chunks, embeddings, metadatas):      # Take text chunks from a document and safely store or update them in ChromaDB
            self.collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )


    def similarity_search(self, query_embedding, top_k, where=None):   # Perform a similarity search in ChromaDB using the query embedding and return the most relevant docs
        results = self.collection.query(
            query_embeddings=[query_embedding],   # Embedding vector for the search query
            n_results=top_k,    # Return the top K most similar documents based on cosine similarity
            where=where         # Optional metadata filter to narrow down search results (e.g., {"source": "document1.pdf"})
        )

        contexts = []
        sources = []

        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        dists = results.get("distances", [])

        for doc, meta in zip(docs[0], metas[0]):
            if doc:
                contexts.append(doc)

                if meta and "source" in meta:
                    sources.append(meta["source"])
                else:
                    sources.append("Unknown")

        
        return {
            "contexts": contexts,
            "sources": list(sources),
            "scores": dists[0] if dists else []
        }

        
ChromaVectorStore()

# client = chromadb.Client(
#     Settings(
#         persist_directory="data/chroma_db",
#         anonymized_telemetry=False,
#         is_persistent=True
#     )
# )

# collection = client.get_or_create_collection(
#     name="docs",
#     metadata={"hnsw:space": "cosine"}
# )

# # Example ingestion
# collection.upsert(
#     ids=["doc1"],
#     documents=["This is a test document."],
#     metadatas=[{"source": "test"}],
#     embeddings=[[0.1, 0.2, 0.3]]  # must match embedding dimension
# )
# # Now this will delete the persisted collection

#print("Existing collections:", client.list_collections())  # Check existing collections after deletion
# recreate collection
# collection = client.create_collection(name="university_docs")

# client.delete_collection(name="docs")