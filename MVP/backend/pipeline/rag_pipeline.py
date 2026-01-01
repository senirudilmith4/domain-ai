"""
Complete RAG Pipeline: Load -> Chunk -> Embed -> Store -> Query
"""

from typing import List, Dict, Any
from MVP.backend.utils.logger import get_logger
from MVP.backend.ingestion.load_docs import load_documents
from MVP.backend.ingestion.chunk_docs import chunk_documents
from MVP.backend.vector_store.embedder import embed_documents
from MVP.backend.vector_store.chroma_client import ChromaVectorStore

logger = get_logger(__name__)


class RAGPipeline:
    """Complete RAG pipeline for document processing and querying."""
    
    def __init__(self, collection_name: str = "documents"):
        """
        Initialize RAG pipeline.
        
        Args:
            collection_name: Name of the ChromaDB collection to use
        """
        self.vector_store = ChromaVectorStore(collection_name=collection_name)
        logger.info(f"RAG Pipeline initialized with collection: {collection_name}")
    
    def ingest_documents(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Complete document ingestion pipeline: Load -> Chunk -> Embed -> Store.
        
        Args:
            chunk_size: Size of text chunks in characters
            overlap: Overlap between chunks in characters
            batch_size: Batch size for embedding generation
        
        Returns:
            Dictionary with ingestion statistics
        """
        try:
            # Step 1: Load documents
            logger.info("Step 1/4: Loading documents...")
            documents = load_documents()
            
            if not documents:
                logger.warning("No documents loaded. Aborting ingestion.")
                return {
                    "success": False,
                    "message": "No documents found to ingest",
                    "documents_loaded": 0,
                    "chunks_created": 0,
                    "chunks_stored": 0
                }
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Step 2: Chunk documents
            logger.info("Step 2/4: Chunking documents...")
            chunks = chunk_documents(
                documents=documents,
                chunk_size=chunk_size,
                overlap=overlap
            )
            
            if not chunks:
                logger.warning("No chunks created. Aborting ingestion.")
                return {
                    "success": False,
                    "message": "No chunks created from documents",
                    "documents_loaded": len(documents),
                    "chunks_created": 0,
                    "chunks_stored": 0
                }
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            logger.info("Step 3/4: Generating embeddings...")
            embeddings = embed_documents(
                chunks=chunks,
                batch_size=batch_size,
                show_progress=True
            )
            
            if not embeddings:
                logger.warning("No embeddings generated. Aborting ingestion.")
                return {
                    "success": False,
                    "message": "Failed to generate embeddings",
                    "documents_loaded": len(documents),
                    "chunks_created": len(chunks),
                    "chunks_stored": 0
                }
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Step 4: Store in vector database
            logger.info("Step 4/4: Storing in vector database...")
            
            # Create metadata for each chunk (optional but recommended)
            metadatas = [
                {"chunk_index": i, "chunk_size": len(chunk)}
                for i, chunk in enumerate(chunks)
            ]
            
            self.vector_store.add_documents(
                chunks=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            # Get final stats
            stats = self.vector_store.get_collection_stats()
            
            logger.info("Document ingestion completed successfully!")
            
            return {
                "success": True,
                "message": "Documents ingested successfully",
                "documents_loaded": len(documents),
                "chunks_created": len(chunks),
                "chunks_stored": len(embeddings),
                "total_in_db": stats.get("total_documents", 0)
            }
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Ingestion failed: {str(e)}",
                "documents_loaded": 0,
                "chunks_created": 0,
                "chunks_stored": 0
            }
    
    def query(
        self,
        query_text: str,
        n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Query the RAG system with a text query.
        
        Args:
            query_text: The question or query text
            n_results: Number of relevant chunks to retrieve
        
        Returns:
            Dictionary with query results and context
        """
        try:
            if not query_text or not query_text.strip():
                logger.warning("Empty query provided")
                return {
                    "success": False,
                    "message": "Query text cannot be empty",
                    "results": []
                }
            
            logger.info(f"Processing query: '{query_text[:100]}...'")
            
            # Step 1: Generate embedding for query
            logger.info("Generating query embedding...")
            query_embeddings = embed_documents(
                chunks=[query_text],
                show_progress=False
            )
            
            if not query_embeddings:
                logger.error("Failed to generate query embedding")
                return {
                    "success": False,
                    "message": "Failed to generate query embedding",
                    "results": []
                }
            
            query_embedding = query_embeddings[0]
            
            # Step 2: Search vector database
            logger.info(f"Searching for top {n_results} relevant chunks...")
            results = self.vector_store.query(
                query_embedding=query_embedding,
                n_results=n_results
            )
            
            # Step 3: Format results
            formatted_results = []
            
            if results and results.get('documents') and results['documents'][0]:
                documents = results['documents'][0]
                distances = results['distances'][0]
                metadatas = results.get('metadatas', [[]])[0]
                ids = results.get('ids', [[]])[0]
                
                for i in range(len(documents)):
                    formatted_results.append({
                        "rank": i + 1,
                        "text": documents[i],
                        "similarity_score": 1 - distances[i],  # Convert distance to similarity
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "id": ids[i] if i < len(ids) else None
                    })
            
            logger.info(f"Query completed. Retrieved {len(formatted_results)} results")
            
            return {
                "success": True,
                "message": "Query completed successfully",
                "query": query_text,
                "num_results": len(formatted_results),
                "results": formatted_results
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Query failed: {str(e)}",
                "results": []
            }
    
    def get_context_for_llm(
        self,
        query_text: str,
        n_results: int = 3,
        max_context_length: int = 2000
    ) -> str:
        """
        Get formatted context from retrieved documents for LLM prompting.
        
        Args:
            query_text: The query to search for
            n_results: Number of chunks to retrieve
            max_context_length: Maximum total length of context
        
        Returns:
            Formatted context string ready for LLM prompt
        """
        query_result = self.query(query_text, n_results=n_results)
        
        if not query_result.get("success") or not query_result.get("results"):
            return "No relevant context found."
        
        context_parts = []
        current_length = 0
        
        for result in query_result["results"]:
            chunk_text = result["text"]
            
            # Check if adding this chunk would exceed max length
            if current_length + len(chunk_text) > max_context_length:
                # Add partial chunk if there's room
                remaining = max_context_length - current_length
                if remaining > 100:  # Only add if meaningful space remains
                    context_parts.append(chunk_text[:remaining] + "...")
                break
            
            context_parts.append(f"[Context {result['rank']}]:\n{chunk_text}")
            current_length += len(chunk_text)
        
        return "\n\n".join(context_parts)
    
    def answer_with_llm(
        self,
        query_text: str,
        n_results: int = 3,
        max_context_length: int = 2000
    ) -> Dict[str, Any]:
        """
        Answer a query using RAG: Retrieve context and generate answer with LLM.
        
        Args:
            query_text: User's question
            n_results: Number of chunks to retrieve for context
            max_context_length: Maximum context length to send to LLM
        
        Returns:
            Dictionary with answer, context, and metadata
        """
        try:
            # Import here to avoid circular dependency
            from MVP.backend.llm.client import LLMClient
            
            logger.info(f"Processing RAG query: {query_text}")
            
            # Step 1: Retrieve relevant context
            context = self.get_context_for_llm(
                query_text=query_text,
                n_results=n_results,
                max_context_length=max_context_length
            )
            
            if context == "No relevant context found.":
                logger.warning("No context retrieved for query")
                return {
                    "success": False,
                    "query": query_text,
                    "answer": "I couldn't find relevant information to answer your question.",
                    "context": "",
                    "sources": []
                }
            
            # Step 2: Generate answer using LLM
            llm = LLMClient()
            answer = llm.ask(question=query_text, context=context)
            
            # Get source information
            query_result = self.query(query_text, n_results=n_results)
            sources = [
                {
                    "text": r["text"][:100] + "...",
                    "similarity": r["similarity_score"]
                }
                for r in query_result.get("results", [])
            ]
            
            logger.info("RAG answer generated successfully")
            
            return {
                "success": True,
                "query": query_text,
                "answer": answer,
                "context": context,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"RAG answer generation failed: {e}", exc_info=True)
            return {
                "success": False,
                "query": query_text,
                "answer": f"Error generating answer: {str(e)}",
                "context": "",
                "sources": []
            }
    
    def clear_collection(self) -> None:
        """Clear all documents from the current collection."""
        logger.warning("Clearing collection...")
        self.vector_store.delete_collection()
        # Reinitialize collection
        self.vector_store._collection = None
        _ = self.vector_store.collection
        logger.info("Collection cleared and reinitialized")


# Convenience functions for direct use
def setup_rag_system(collection_name: str = "documents") -> RAGPipeline:
    """
    Initialize and return a RAG pipeline.
    
    Args:
        collection_name: Name of the collection to use
    
    Returns:
        Initialized RAGPipeline instance
    """
    return RAGPipeline(collection_name=collection_name)


def ingest_documents_to_rag(
    pipeline: RAGPipeline,
    chunk_size: int = 500,
    overlap: int = 100
) -> Dict[str, Any]:
    """
    Ingest documents using the RAG pipeline.
    
    Args:
        pipeline: RAGPipeline instance
        chunk_size: Size of text chunks
        overlap: Overlap between chunks
    
    Returns:
        Ingestion statistics
    """
    return pipeline.ingest_documents(
        chunk_size=chunk_size,
        overlap=overlap
    )


def query_rag_system(
    pipeline: RAGPipeline,
    query: str,
    n_results: int = 5
) -> Dict[str, Any]:
    """
    Query the RAG system.
    
    Args:
        pipeline: RAGPipeline instance
        query: Query text
        n_results: Number of results to return
    
    Returns:
        Query results
    """
    return pipeline.query(query_text=query, n_results=n_results)


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    rag = setup_rag_system(collection_name="my_documents")
    
    # Ingest documents
    print("Starting document ingestion...")
    ingestion_stats = ingest_documents_to_rag(rag, chunk_size=500, overlap=100)
    print(f"\nIngestion Stats: {ingestion_stats}")
    
    # Query the system
    print("\n" + "="*50)
    print("Querying the RAG system...")
    query = "What is the main topic of these documents?"
    results = query_rag_system(rag, query, n_results=3)
    
    print(f"\nQuery: {query}")
    print(f"Found {results['num_results']} relevant chunks:\n")
    
    for result in results.get('results', []):
        print(f"Rank {result['rank']} (Similarity: {result['similarity_score']:.3f})")
        print(f"{result['text'][:200]}...")
        print("-" * 50)