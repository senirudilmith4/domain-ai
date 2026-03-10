import logging
import inngest
import inngest.fast_api  # Automatically exposes Inngest endpoints inside FastAPI app
import os
import datetime
import uuid
from fastapi import FastAPI, HTTPException, Header
from inngest.experimental import ai  # Simplifies calling LLMs, managing retries, background AI execution
from dotenv import load_dotenv
from torch import chunk
from pathlib import Path

# from app.api.routes.ask import router as ask_router
from ingestion.load_docs import DOCS_PATH, load_and_chunk_pdf, embed_texts
from chroma_db.vector_db import ChromaVectorStore
from app.schemas.custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult
from app.OllamaAdapter import OllamaAdapter

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
        pdf_files = list(DOCS_PATH.glob("*.pdf"))  # List all PDF files in the documents directory
        if not pdf_files:
            raise ValueError(f"No PDF files found in {DOCS_PATH}")
        all_chunks = []
        sources = []

        for pdf in pdf_files:
            chunks = load_and_chunk_pdf(str(pdf))
            pdf_name = pdf.name

            for c in chunks:
                chunk_with_source = f"Module: {pdf_name}\n\n{c}"

                all_chunks.append(chunk_with_source)
                sources.append(pdf_name)

        return RAGChunkAndSrc(
            chunks=all_chunks,
            sources=sources
)
    def _upsert(chunk_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunk_and_src.chunks
        sources = chunk_and_src.sources

        vecs = embed_texts(chunks)

        ids = [str(uuid.uuid4()) for _ in chunks]

        payloads = [
            {
                "source": sources[i],
                "text": chunks[i]
            }
            for i in range(len(chunks))
        ]  # Create payloads containing the source ID and corresponding text chunk
        ChromaVectorStore().upsert(ids,chunks,vecs, payloads)  # Upsert the embeddings and payloads into the Chroma vector store
        return RAGUpsertResult(ingested=len(chunks))  # Return the number of chunks ingested as part of the result
    
    chunk_and_src = await ctx.step.run("load-and-chunk", lambda:_load(ctx), output_type=RAGChunkAndSrc)  # Run the loading and chunking step, passing the context and specifying the output type
    ingested = await ctx.step.run("embed-and-upsert", lambda:_upsert(chunk_and_src), output_type=RAGUpsertResult)  # Run the embedding and upserting step, passing the chunk and source data, and specifying the output type
    print("Chunks ingested:", len(chunk_and_src.chunks))  # Log the number of chunks ingested
    return ingested.model_dump()  # Return the result of the ingestion process as a dictionary
    

@inngest_client.create_function(
    fn_id= "RAG: Query PDF",  # Unique identifier for the function
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")  # Specifies the event that triggers this function
 )
async def query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int =5)-> RAGSearchResult:
        query_vec = embed_texts([question])[0]  # Converts user question into a vector embedding
        store = ChromaVectorStore()  # Creates connection to Vector DB
        found = store.similarity_search(query_vec, top_k)  # Retrieves top_k most similar chunks
        return RAGSearchResult(contexts=found['contexts'], sources=found['sources'])  # Wrap into structured result
    
    question = ctx.event.data["question"]  # Get the question from the event data
    top_k = ctx.event.data.get("top_k", 5)  # Get the top_k parameter from the event data, defaulting to 5
    
    found = await ctx.step.run('embed-and-search', lambda: _search(question,top_k), output_type=RAGSearchResult)  # Run the embedding and searching step, passing the question and top_k parameters, and specifying the output type
    context_block = "\n\n".join(
    f"[Source: {found.sources[i]}]\n{found.contexts[i]}"
    for i in range(len(found.contexts))
    )  # Combine the found contexts into a single block of text
    system_prompt = """
        You are a University Academic Assistant.

        Answer ONLY using the provided context from university documents.

        If the answer is not found in the context, respond exactly with:
        "I could not find this in the university documents."

        Always format answers using clean Markdown.

        Formatting rules:
        - Use headings (###) for sections
        - Use numbered lists for learning outcomes or steps
        - Use bullet points for explanations
        - Leave blank lines between sections

        Response format:

        ### Answer
        A short explanation answering the question.

        ### Details
        Structured information such as outcomes, steps, or definitions.

        ### Sources
        List the document names where the information was found.
        """

    prompt = f"""
    {system_prompt}

    --------------------------------
    Context from University Documents:
    {context_block}
    --------------------------------

    Student Question:
    {question}

    Write the answer now.
    """

    answer = await ctx.step.run(
        "llm-answer",
        lambda: OllamaAdapter().generate(prompt),  # Call the OllamaAdapter to generate an answer based on the system prompt and the combined context
    )
    print("Query found contexts:", found.contexts)  # Log the contexts found during the search
    return {
    "answer": answer.strip(),
    "sources": list(set(found.sources)),
    "num_contexts": len(found.contexts)
  }

inngest.fast_api.serve(app,inngest_client,[ingest_document,query_pdf_ai])  # Bridge Inngest with FastAPI, enabling serverless function execution

@app.get("/health")
def health_check():
    return {"status": "backend is running"} # defines a simple endpoint to check if the backend is running. When you access /health, it will return a JSON response indicating the status of the backend.


#  { "data":{
#    "question": "What are the Learning Outcomes On successful completion of this Module 2607 "
#    }
#  }


# { "data":{
#    "pdf_path": "D:\\OneDrive\\Documents\\IIT\\STAGE 02\\DSGP\\Domain AI\\MVP\\backend\\data\\docs\\Module CM2607 Advanced Mathematics for Data Science.pdf"
#    }
#  }



# rouge , bleu testing, f1 score llm testing