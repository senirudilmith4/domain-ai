import asyncio
import logging
import inngest
import inngest.fast_api  # Automatically exposes Inngest endpoints inside FastAPI app
import uuid
from fastapi import FastAPI, HTTPException, Header
from inngest.experimental import ai  # Simplifies calling LLMs, managing retries, background AI execution
from dotenv import load_dotenv

# from app.api.routes.ask import router as ask_router
from ingestion.load_docs import DOCS_PATH, load_pdf,chunk_texts, embed_texts,sample_texts_for_metadata,chunk_id
from chroma_db.vector_db import ChromaVectorStore
from app.schemas.custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult
from app.OllamaAdapter import OllamaAdapter
from app.schemas.meta_detec import detect_doc_type_and_metadata, extract_filters

load_dotenv()  # Load environment variables from .env file
logger = logging.getLogger("uvicorn")

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
vector_db = ChromaVectorStore()
# app.include_router(ask_router, prefix="/api")   # take all the routes defined in ask router and add them to the main app with the prefix /api

@inngest_client.create_function(
    fn_id="RAG: Ingest Document",
    trigger=inngest.TriggerEvent(event="rag/inngest-document"),
)
async def ingest_document(ctx: inngest.Context):
 
    def _load() -> RAGChunkAndSrc:
        """Load every PDF in DOCS_PATH, chunk it, and collect metadata."""
        pdf_files = list(DOCS_PATH.rglob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {DOCS_PATH}")
 
        all_chunks: list[str] = []
        all_ids: list[str] = []
        metadatas: list[dict] = []
 
        for pdf in pdf_files:
            source = str(pdf)
            texts = load_pdf(source)
 
            # Sample across the full document for better type detection
            sample = sample_texts_for_metadata(texts)
            base_metadata = detect_doc_type_and_metadata(source, sample)
            doc_type = base_metadata.get("doc_type", "default")
 
            # Use doc-type-aware chunking
            chunks = chunk_texts(texts, doc_type=doc_type)
 
            for i, c in enumerate(chunks, start=1):
                all_chunks.append(c)
                # Deterministic ID prevents duplicate vectors on re-ingestion
                all_ids.append(chunk_id(c, source))
                metadatas.append({**base_metadata, "chunk": i})
 
        return RAGChunkAndSrc(chunks=all_chunks, metadatas=metadatas)
 
    def _upsert(chunk_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        """Embed chunks and upsert into ChromaDB."""
        chunks = chunk_and_src.chunks
        vecs = embed_texts(chunks)
        # Re-derive deterministic IDs from the chunk content + source stored in metadata
        ids = [
            chunk_id(c, meta.get("source", "unknown"))
            for c, meta in zip(chunks, chunk_and_src.metadatas)
        ]
        vector_db.upsert(ids, chunks, vecs, chunk_and_src.metadatas)
        return RAGUpsertResult(ingested=len(chunks))
 
    chunk_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: asyncio.to_thread(_load),
        output_type=RAGChunkAndSrc,
    )
 
    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: asyncio.to_thread(_upsert, chunk_and_src),
        output_type=RAGUpsertResult,
    )
 
    logger.info("Chunks ingested: %d", ingested.ingested)
    return ingested.model_dump()
    


def _build_where_filter(filters: dict | None) -> dict | None:
    """
    Safely convert an extracted filter dict into a ChromaDB `where` clause.
 
    ChromaDB requires that every field used in a `where` clause exists on ALL
    documents in the collection — otherwise it raises an error. We guard
    against unknown fields here so a bad LLM extraction never crashes the query.
 
    Supported fields: doc_type
    """
    if not filters or not isinstance(filters, dict):
        return None
 
    allowed_fields = {"doc_type"}
    valid = {k: v for k, v in filters.items() if k in allowed_fields and isinstance(v, str)}
 
    if not valid:
        return None
 
    # Single-field filter: {"doc_type": "policy"}
    if len(valid) == 1:
        field, value = next(iter(valid.items()))
        return {field: {"$eq": value}}
 
    # Multi-field filter: wrap in $and
    return {"$and": [{k: {"$eq": v}} for k, v in valid.items()]}
 
 
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def query_pdf_ai(ctx: inngest.Context):
 
    def _search(question: str, top_k: int, filters: dict | None) -> RAGSearchResult:
        """
        Two-pass retrieval:
        1. Filtered search — if filters are available, narrow the search to
           matching doc_type first (faster, more precise).
        2. Fallback to unfiltered search if the filtered pass returns too few
           results (e.g. the filter was too strict or doc wasn't ingested with
           that metadata yet).
        """
        query_embedding = embed_texts([question])[0]
        where_clause = _build_where_filter(filters)
 
        results = None
 
        # Pass 1: filtered
        if where_clause:
            try:
                results = vector_db.similarity_search(query_embedding, top_k, where=where_clause)
                logger.info("Filtered search with %s returned %d results", where_clause, len(results["contexts"]))
            except Exception as e:
                # ChromaDB throws if the field doesn't exist on any doc — fall through
                logger.warning("Filtered search failed (%s) — falling back to unfiltered", e)
                results = None
 
        # Pass 2: unfiltered fallback
        if not results or len(results["contexts"]) < 2:
            if where_clause:
                logger.info("Filtered pass returned too few results — running unfiltered fallback")
            results = vector_db.similarity_search(query_embedding, top_k, where=None)
 
        for i, (ctx_text, src) in enumerate(zip(results["contexts"], results["sources"])):
            logger.info("Result %d | source: %s | preview: %s", i + 1, src, ctx_text[:200])
 
        return RAGSearchResult(contexts=results["contexts"], sources=results["sources"])
 
    # ---- pull inputs from event ----
    question: str = ctx.event.data["question"]
    top_k: int = ctx.event.data.get("top_k", 10)
 
    # Step 1: extract metadata filters from the question
    filters = await ctx.step.run(
        "extract-filters",
        lambda: asyncio.to_thread(extract_filters, question),
    )
    logger.info("Extracted filters: %s", filters)
 
    # Step 2: embed + filtered semantic search
    found = await ctx.step.run(
        "embed-and-search",
        lambda: asyncio.to_thread(_search, question, top_k, filters),
        output_type=RAGSearchResult,
    )
 
    # Step 3: build prompt with retrieved context
    context_block = "\n\n".join(
        f"[Source: {found.sources[i]}]\n{found.contexts[i]}"
        for i in range(len(found.contexts))
    )
 
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
        """.strip()
 
    prompt = f"""
        {system_prompt}
        
        --------------------------------
        Context from University Documents:
        {context_block}
        --------------------------------
        
        Student Question:
        {question}
        
        Write the answer now.
        """.strip()
 
    # Step 4: generate answer via Ollama
    answer = await ctx.step.run(
        "llm-answer",
        lambda: asyncio.to_thread(OllamaAdapter().generate, prompt),
    )
 
    logger.info("Query complete — %d contexts used", len(found.contexts))
 
    return {
        "answer": answer.strip(),
        "sources": list(set(found.sources)),
        "num_contexts": len(found.contexts),
        "filters_applied": filters,
        "contexts": found.contexts,
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