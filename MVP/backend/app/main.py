import logging
import inngest
import inngest.fast_api  # Automatically exposes Inngest endpoints inside FastAPI app
import os
import datetime
import uuid
from fastapi import FastAPI, HTTPException, Header
from inngest.experimental import ai  # Simplifies calling LLMs, managing retries, background AI execution
from dotenv import load_dotenv
# from torch import chunk
# from pathlib import Path

# from app.api.routes.ask import router as ask_router
from ingestion.load_docs import DOCS_PATH, load_pdf,chunk_texts, embed_texts
from chroma_db.vector_db import ChromaVectorStore
from app.schemas.custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult
from app.OllamaAdapter import OllamaAdapter
from app.schemas.meta_detec import detect_doc_type_and_metadata

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
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc: # Load and chunk the PDF document
        pdf_files = list(DOCS_PATH.glob("*.pdf"))  # List all PDF files in the documents directory
        if not pdf_files:
            raise ValueError(f"No PDF files found in {DOCS_PATH}")
        all_chunks = []
        metadatas= []

        for pdf in pdf_files:
            texts = load_pdf(str(pdf))  # Load the PDF and extract text
            chunks = chunk_texts(texts)  # Chunk the extracted text into smaller pieces
            sample_texts = " ".join(texts[:2])  # Take only the first 2 chunks for metadata detection
            base_metadata = detect_doc_type_and_metadata(str(pdf), sample_texts)  # Extract metadata from the sample text
            for i,c in enumerate(chunks,start=1):
                all_chunks.append(c)
                metadata ={**base_metadata, "chunk": i}  # Add chunk index to metadata
                metadatas.append(metadata)

        return RAGChunkAndSrc(
            chunks=all_chunks,
            metadatas=metadatas)
    
    def _upsert(chunk_and_src: RAGChunkAndSrc) -> RAGUpsertResult: # Embed the chunks and upsert them into the vector database
        chunks = chunk_and_src.chunks
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = chunk_and_src.metadatas
        ChromaVectorStore().upsert(ids,chunks,vecs, metadatas)  # Upsert the embeddings and metadatas into the Chroma vector store
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
        query_embedding = embed_texts([question])[0]  # Embed the question to get its vector representation for similarity search
        store = ChromaVectorStore()  # Creates connection to Vector DB
        found = store.similarity_search(query_embedding, 10)  # Retrieves top_k most similar chunks
        store = ChromaVectorStore()
        results = store.similarity_search(query_embedding, 20)

        for i in range(len(results["contexts"])):
            print("SOURCE:", results["sources"][i])
            print(results["contexts"][i][:200])
            print("------")
        return RAGSearchResult(contexts=found['contexts'], sources=found['sources'])  # Wrap into structured result
    
    question = ctx.event.data["question"]  # Get the question from the event data
    top_k = ctx.event.data.get("top_k", 10)  # Get the top_k parameter from the event data, defaulting to 5
    
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
    "num_contexts": len(found.contexts),
    "Query found contexts:": found.contexts

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

# Traditional NLP Metrics
    # BLEU,ROUGE-L,F1 Score

# RAG-Specific Metrics
    # Retrieval accuracy, Context relevance, Hallucination rate

# Example statement in thesis:
# The system was evaluated using BLEU, ROUGE-L, and F1 scores to measure textual similarity between generated responses and reference answers. 
# Additionally, retrieval accuracy and hallucination tests were conducted to assess the effectiveness of the RAG pipeline.