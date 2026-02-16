import logging
import inngest
import inngest.fast_api  # Automatically exposes Inngest endpoints inside FastAPI app
import os
import datetime
import uuid
from fastapi import FastAPI, HTTPException, Header
from inngest.experimental import ai  # Simplifies calling LLMs, managing retries, background AI execution
from dotenv import load_dotenv
# from app.api.routes.ask import router as ask_router
from ingestion.load_docs import load_and_chunk_pdf, embed_texts
from chroma_db.vector_db import ChromaVectorStore
from app.custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult

load_dotenv()  # Load environment variables from .env file

inngest_client = inngest.Inngest(   # Initialize Inngest client for handling serverless functions and AI tasks
    app_id="rag_api",               # Unique identifier for Inngest application
    logger = logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)
app = FastAPI(                # creates a web server
    title="LLM-RAG API", 
    version="1.0.0"
    )     

# app.include_router(ask_router, prefix="/api")   # take all the routes defined in ask router and add them to the main app with the prefix /api

@inngest_client.create_function(
    fn_id= "RAG: Ingest Document",  # Unique identifier for the function
    trigger=inngest.TriggerEvent(event="rag/inngest-document")  # Specifies the event that triggers this function
)
async def ingest_document(ctx: inngest.Context): 
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]  # Get the PDF file path from the event data
        source_id = ctx.event.data.get("source_id", pdf_path)  # Use source_id from event data if provided, otherwise default to pdf_path
        chunks = load_and_chunk_pdf(pdf_path)  # Load and chunk the PDF document
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)  # Return the chunks along with the source ID 
    
    def _upsert(chunk_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunk_and_src.chunks  # Get the list of text chunks from the RAGChunkAndSrc object
        source_id = chunk_and_src.source_id  # Get the source ID associated with the chunks
        vecs = embed_texts(chunks)  # Generate vector embeddings for the chunks of text
        ids =[str(uuid.uuid5(uuid.NAMESPACE_URL,f"{source_id}:{i}")) for i in range(len(chunks))]  # Create unique IDs for each chunk using UUID5 based on the source ID and chunk index
        payloads = [{"source": source_id, "text": chunks[i]} for i in range (len(chunks))]  # Create payloads containing the source ID and corresponding text chunk
        ChromaVectorStore().upsert(ids,chunks,vecs, payloads)  # Upsert the embeddings and payloads into the Chroma vector store
        return RAGUpsertResult(ingested=len(chunks))  # Return the number of chunks ingested as part of the result
    
    chunk_and_src = await ctx.step.run("load-and-chunk", lambda:_load(ctx), output_type=RAGChunkAndSrc)  # Run the loading and chunking step, passing the context and specifying the output type
    ingested = await ctx.step.run("embed-and-upsert", lambda:_upsert(chunk_and_src), output_type=RAGUpsertResult)  # Run the embedding and upserting step, passing the chunk and source data, and specifying the output type
    return ingested.model_dump()  # Return the result of the ingestion process as a dictionary
      

inngest.fast_api.serve(app,inngest_client,[ingest_document])  # Bridge Inngest with FastAPI, enabling serverless function execution


@inngest_client.create_function(
    fn_id= "RAG: Query PDF",  # Unique identifier for the function
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai") 
 ) # Specifies the event that triggers this function

async def query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int =5)-> RAGSearchResult:
        query_vec = embed_texts([question])[0]  # Converts user question into a vector embedding
        store = ChromaVectorStore()  # Creates connection to Vector DB
        found = store.search(query_vec, top_k)  # Retrieves top_k most similar chunks
        return RAGSearchResult(contexts=found['contexts'], sources=found['sources'])  # Wrap into structured result
    
    question = ctx.event.data["question"]  # Get the question from the event data
    top_k = ctx.event.data.get("top_k", 5)  # Get the top_k parameter from the event data, defaulting to 5
    
    found = await ctx.step.run('embed-and-search', lambda: _search(question,top_k), output_type=RAGSearchResult)  # Run the embedding and searching step, passing the question and top_k parameters, and specifying the output type
    context_block = "\n\n".join(f" -{c}" for c in found.contexts)  # Combine the found contexts into a single block of text
    user_content = (
        "Use the following retrieved contexts to answer the question. If you don't know the answer, say you don't know.\n\n"
        f"Context: \n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely based on the provided contexts."  # Create the prompt for the LLM
    )

    
    answer = await ctx.step.run(
        "llm-answer",
        lambda: OllamaAdapter().generate(user_content)
    )

    return {
        "answer": answer.strip(),
        "sources": found.sources,
        "num_contexts": len(found.contexts)
    }

@app.get("/health")
def health_check():
    return {"status": "backend is running"} # defines a simple endpoint to check if the backend is running. When you access /health, it will return a JSON response indicating the status of the backend.


